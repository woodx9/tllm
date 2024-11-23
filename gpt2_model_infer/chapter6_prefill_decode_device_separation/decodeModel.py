import math
import torch
from torch import nn
from transformers.activations import gelu_new
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from utils import getLastPosition
from logger import logger, LogLevel


GPT2_CONFIG_PATH = './../model/config.json'
GPT2_PYTORCH_BIN_PATH = './../model/gpt2_pytorch_model.bin'
DEFAULT_DEVICE = 'cpu'


class DecodeModel(nn.Module):
    def __init__(self, device1=DEFAULT_DEVICE, device2=DEFAULT_DEVICE):
        super().__init__()

        self.device = device1
        self.returnDevice = device2

        # 加载配置和模型参数
        self.config = GPT2Config.from_pretrained(GPT2_CONFIG_PATH)
        self.modelParameters  = torch.load(GPT2_PYTORCH_BIN_PATH, map_location=torch.device(self.device), weights_only=True)
        

        self.embed_dim = self.config.n_embd
        self.wte = nn.Embedding(self.config.vocab_size, self.embed_dim, device=self.device)
        self.wpe = nn.Embedding(self.config.max_position_embeddings, self.embed_dim, device=self.device)


        # self.h = nn.ModuleList([MyGPT2Block(self.config, self.modelParameters, layer_idx=i) for i in range(1)])
        self.h = nn.ModuleList([DecodeGPT2Block(self.config, self.modelParameters, layer_idx=i, device=self.device) for i in range(self.config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=self.config.layer_norm_epsilon, device=self.device)

        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False, device=self.device)
        self.load_model_parameters()

    def load_model_parameters(self):
        self.wte.weight =  nn.Parameter(self.modelParameters['wte.weight'])
        
        self.wpe.weight =  nn.Parameter(self.modelParameters['wpe.weight'])

        self.lm_head.weight = self.wte.weight

        self.ln_f.weight =  nn.Parameter(self.modelParameters['ln_f.weight'])
        self.ln_f.bias =  nn.Parameter(self.modelParameters['ln_f.bias'])

        # logger.info('wte', self.wte.weight)
        # logger.info('wpe', self.wpe.weight)


    def forward(self, input_ids: torch.Tensor, kvCache: torch.Tensor, attention_mask: torch.Tensor):
        logger.info("MyGPT2Model forward: ")
        input_ids = input_ids.to(self.device)

        for i, (key, value) in enumerate(kvCache):
            key = key.to(self.device)
            value = value.to(self.device)
            kvCache[i] = (key, value)

        attention_mask = attention_mask.to(self.device)

        
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=self.device)

        logger.info('input_ids:', input_ids)
        logger.info('position_ids: ', position_ids)

        # 获取token对应的embedding
        inputs_embeds = self.wte(input_ids)
        # 获取token pos对应的embedding
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        ## 这里需要用attention mask匹配所有prompt的最后一个位置
        last_positions = getLastPosition(attention_mask)

        hidden_states = hidden_states[torch.arange(len(last_positions)), last_positions]

        hidden_states = hidden_states.unsqueeze(1)

        newKvCache = []

        # 数据送入GPT2Block中
        for layer in self.h:
            hidden_states, kv = layer(hidden_states, kvCache, attention_mask)
            newKvCache.append(kv)

        hidden_states = self.ln_f(hidden_states)

        # 获取多语言头 
        logits = self.lm_head(hidden_states)
        logger.info('self.lm_head(hidden_states)', logits.shape)

        last_token_logits = logits[:, -1, :]

        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(last_token_logits, dim=-1)

        max_prob_index = torch.argmax(probabilities, dim=-1)
        logger.info('probabilities_np', max_prob_index)

        # newKvCache的数据转移到另一个设备
        for i, (key, value) in enumerate(newKvCache):
            key = key.to(self.returnDevice)
            value = value.to(self.returnDevice)
            newKvCache[i] = (key, value)


        return max_prob_index, newKvCache



class DecodeGPT2Block(nn.Module):
    def __init__(self, config, modelParameters, layer_idx=None, device=DEFAULT_DEVICE):
        super().__init__()
        self.modelParameters = modelParameters
        self.config = config
        self.embed_dim = self.config.n_embd
        self.device = device

        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon, device=self.device)
        self.layer_idx = layer_idx
        self.attn = DecodeGPT2Attention(config, modelParameters, layer_idx=layer_idx, device=self.device)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon, device=self.device)
        self.mlp = MyGPT2MLP(inner_dim, config, layer_idx, modelParameters, device=self.device)

        self.load_model_parameters()

    def load_model_parameters(self):
        # 获取 ln_1 和 ln_2 的参数
        # h.0.ln_1.weight
        ln_1_ParaName = f'h.{self.layer_idx}.ln_1'
        self.ln_1.weight = nn.Parameter(self.modelParameters[f'{ln_1_ParaName}.weight'])
        self.ln_1.bias = nn.Parameter(self.modelParameters[f'{ln_1_ParaName}.bias'])

        ln_2_ParaName = f'h.{self.layer_idx}.ln_2' 
        self.ln_2.weight = nn.Parameter(self.modelParameters[f'{ln_2_ParaName}.weight'])
        self.ln_2.bias = nn.Parameter(self.modelParameters[f'{ln_2_ParaName}.bias'])

        # logger.info(ln_1_ParaName, self.ln_1.weight)
        # logger.info(ln_1_ParaName, self.ln_1.bias)
        # logger.info(ln_2_ParaName, self.ln_2.weight)
        # logger.info(ln_2_ParaName, self.ln_2.bias)


    def forward(self, hidden_states, kvCache, attention_mask):
        logger.info("MyGPT2Block forward: ")
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_out, newKvCache = self.attn(hidden_states, kvCache, attention_mask)
        hidden_states = attn_out + residual


        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states, newKvCache


class DecodeGPT2Attention(nn.Module):
    def __init__(self, config, modelParameters, is_cross_attention=False, layer_idx=None, device=DEFAULT_DEVICE):
        super().__init__()
        self.modelParameters = modelParameters
        self.config = config
        self.embed_dim = config.n_embd
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.layer_idx = layer_idx
        self.device = device

        self.attn  = nn.Linear(self.config.n_ctx, self.config.n_ctx, bias=True, device=self.device)
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True, device=self.device)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True, device=self.device)

        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.load_model_paramters()

    def load_model_paramters(self):
        attn_bias_ParaName = f'h.{self.layer_idx}.attn.bias'
        self.attn.weight = nn.Parameter(torch.ones(self.config.n_ctx, self.config.n_ctx))
        self.attn.bias = nn.Parameter(self.modelParameters[attn_bias_ParaName].squeeze(0).squeeze(0))
        
        c_attn_ParaName = f'h.{self.layer_idx}.attn.c_attn' 
        self.c_attn.weight = nn.Parameter(self.modelParameters[f'{c_attn_ParaName}.weight'].T)
        self.c_attn.bias = nn.Parameter(self.modelParameters[f'{c_attn_ParaName}.bias'])

        c_proj_ParaName = f'h.{self.layer_idx}.attn.c_proj' 
        self.c_proj.weight = nn.Parameter(self.modelParameters[f'{c_proj_ParaName}.weight'].T)
        self.c_proj.bias = nn.Parameter(self.modelParameters[f'{c_proj_ParaName}.bias'])

        # logger.info(attn_bias_ParaName, self.attn.bias)
        # logger.info(c_attn_ParaName, self.c_attn.weight)
        # logger.info(c_attn_ParaName, self.c_attn.bias)
        # logger.info(c_proj_ParaName, self.c_proj.weight)
        # logger.info(c_proj_ParaName, self.c_proj.bias)



    # 这里的hidden_states是一个向量
    def forward(self, hidden_states, kvCache, attention_mask):
        logger.info("MyGPT2Attention forward: ")


        batch_size, _, embed_dim = hidden_states.size()

        # 计算序列长度
        seq_len = kvCache[0][0].size(1)
        logger.info('seq_len:', seq_len)
        
        # 生成 Q, K, V
        # hidden_states_reshaped： [batch_size * 1, embed_dim]
        hidden_states_reshaped = hidden_states.view(-1, hidden_states.size(-1))  

        # hidden_states_reshaped: [batch_size * 1, embed_dim]
        # self.c_attn:  [embed_dim, 3 * embed_dim]
        # qkv:  [batch_size * 1, 3 * embed_dim]
        qkv = self.c_attn(hidden_states_reshaped)  

        # qkv拆分
        # query: [batch_size * 1, embed_dim]
        # key: [batch_size * 1, embed_dim]
        # value: [batch_size * 1, embed_dim]
        query, key, value = torch.split(qkv, self.embed_dim, dim=-1)

        # 需要将储存在key value里的tensor拼接回来
        # 由于在多batch情况下，需要在尾部先拼接上新的key value（都是默认值）
        new_key_column = torch.full((batch_size, 1, embed_dim), 0, device=self.device)
        new_value_column = torch.full((batch_size, 1, embed_dim), 0, device=self.device)

        # new_keys: [batch_size, num_heads, seq_len + 1, embed_dim]
        # new_values: [batch_size, number_heads, seq_len + 1, embed_dim]
        logger.info('kvCache:', kvCache[self.layer_idx][0])
        logger.info('new_key_column:', new_key_column)
        logger.info('device:', self.device)

        new_keys = torch.cat((kvCache[self.layer_idx][0], new_key_column), dim=1)
        new_values = torch.cat((kvCache[self.layer_idx][1], new_value_column), dim=1)


        # 根据对应的mask，将新的key value放到对应的位置
        last_positions = getLastPosition(attention_mask)

        for index, element in enumerate(last_positions):
            new_keys[index, element, :] = key[index]
            new_values[index, element, :] = value[index]
        

        key = new_keys
        value = new_values

        kvCache = (key, value)
        seq_len = seq_len + 1

         
        # 重塑 Q, K, V 以适应多头
        # query: [batch_size, num_heads, 1, head_dim]
        # key: [batch_size, num_heads, seq_len, head_dim]
        # value: [batch_size, num_heads, seq_len, head_dim]
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        
        # 计算注意力得分
        # Q @ K^T
        # query: [batch_size, num_heads, 1, head_dim]
        # key.transpose(-2, -1): [batch_size, num_heads, head_dim, seq_len]
        # scores: [batch_size, num_heads, 1, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用偏置
        bias = self.attn.bias[:1, :seq_len]
        bias = bias.unsqueeze(0).unsqueeze(0)
        bias = bias.expand(batch_size, self.num_heads, 1, seq_len)

        scores = scores + bias
 
        # 需要屏蔽当前batch的seq_len之后的scores
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(extended_attention_mask == 0, -1e9)
        logger.info('scores: ', scores.shape)
        
        # 计算注意力权重
        # scores: [batch_size, num_heads, 1, seq_len]
        # attn_weights: [batch_size, num_heads, 1, seq_len]
        attn_weights = torch.softmax(scores, dim=-1)  
        
        # 应用注意力权重到 V
        # attn_weights: [batch_size, num_heads, 1, seq_len]
        # value: [batch_size, num_heads, seq_len, head_dim]
        # context: [batch_size, num_heads, 1, head_dim]
        context = torch.matmul(attn_weights, value)
        
        # 合并多头
        # context: [batch_size, num_heads, 1, head_dim]
        # 转换为 [batch_size, 1, num_heads * head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, embed_dim)

        # 最终的线性变换
        # [batch_size, 1, embed_dim]
        context = self.c_proj(context)  

        output = self.resid_dropout(context)
        logger.info('output.shape: ', output.shape)

        return output, kvCache




class MyGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config, layer_idx, modelParameters, device=DEFAULT_DEVICE):
        super().__init__()
        self.embed_dim = config.n_embd
        self.modelParameters = modelParameters
        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        self.c_fc = nn.Linear(intermediate_size, self.embed_dim, bias=True, device=self.device)
        self.c_proj = nn.Linear(self.embed_dim, intermediate_size, bias=True, device=self.device)

        ACT2FN = {
            "gelu": nn.functional.gelu,
            "relu": nn.functional.relu,
            "gelu_new": gelu_new,
        }
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

        self.load_model_parameters()

    def load_model_parameters(self):
        c_fc_ParaName = f'h.{self.layer_idx}.mlp.c_fc' 
        self.c_fc.weight = nn.Parameter(self.modelParameters[f'{c_fc_ParaName}.weight'].T)
        self.c_fc.bias = nn.Parameter(self.modelParameters[f'{c_fc_ParaName}.bias'])

        c_proj_ParaName = f'h.{self.layer_idx}.mlp.c_proj' 
        self.c_proj.weight = nn.Parameter(self.modelParameters[f'{c_proj_ParaName}.weight'].T)
        self.c_proj.bias = nn.Parameter(self.modelParameters[f'{c_proj_ParaName}.bias'])

        # logger.info(c_fc_ParaName, self.c_fc.weight)
        # logger.info(c_fc_ParaName, self.c_fc.bias)
        # logger.info(c_proj_ParaName, self.c_proj.weight)
        # logger.info(c_proj_ParaName, self.c_proj.bias)


    def forward(self, hidden_states) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    

