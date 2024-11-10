import math
import torch
from torch import nn
from transformers.activations import gelu_new
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
import time
import matplotlib.pyplot as plt
import numpy as np
from logger import logger, LogLevel


GPT2_CONFIG_PATH = './../model/config.json'
GPT2_PYTORCH_BIN_PATH = './../model/gpt2_pytorch_model.bin'
DEFAULT_DEVICE = 'cpu'


class PrefillModel(nn.Module):
    def __init__(self, device1=DEFAULT_DEVICE, device2=DEFAULT_DEVICE):
        super().__init__()

        self.device1 = device1
        self.device2 = device2

        # 加载配置和模型参数
        self.config = GPT2Config.from_pretrained(GPT2_CONFIG_PATH)
        self.modelParameters  = torch.load(GPT2_PYTORCH_BIN_PATH, map_location=torch.device(self.device1), weights_only=True)
        modelParametersDevice2  = torch.load(GPT2_PYTORCH_BIN_PATH, map_location=torch.device(self.device2), weights_only=True)

        self.embed_dim = self.config.n_embd
        self.wte = nn.Embedding(self.config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(self.config.max_position_embeddings, self.embed_dim)

        self.h = nn.ModuleList([PrefillGPT2Block(self.config, modelParametersDevice2, device=self.device2, layer_idx=i) for i in range(self.config.num_hidden_layers)])

        self.ln_f = nn.LayerNorm(self.embed_dim, eps=self.config.layer_norm_epsilon)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        

        self.load_model_parameters()

    def load_model_parameters(self):
        self.wte.weight =  nn.Parameter(self.modelParameters['wte.weight'])
        
        self.wpe.weight =  nn.Parameter(self.modelParameters['wpe.weight'])

        self.lm_head.weight = self.wte.weight

        self.ln_f.weight =  nn.Parameter(self.modelParameters['ln_f.weight'])
        self.ln_f.bias =  nn.Parameter(self.modelParameters['ln_f.bias'])

        # logger.info('wte', self.wte.weight)
        # logger.info('wpe', self.wpe.weight)


    def forward(self, input_ids: torch.Tensor):
        logger.info("MyGPT2Model forward: ")
        input_ids = input_ids.to(self.device1)
        
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=self.device1)
        logger.info('input_ids:', input_ids)
        logger.info('position_ids: ', position_ids)

        # 获取token对应的embedding
        inputs_embeds = self.wte(input_ids)
        # 获取token pos对应的embedding
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        kvCache = []

        # 数据转给device2
        hidden_states = hidden_states.to(self.device2)

        # 数据送入GPT2Block中
        for layer in self.h:
            hidden_states, kv = layer(hidden_states)
            kvCache.append(kv)

        # 数据传到device1
        hidden_states = hidden_states.to(self.device1)

        hidden_states = self.ln_f(hidden_states)

        # 获取多语言头 
        logits = self.lm_head(hidden_states)

        # Get logits for the last token in the input sequence
        last_token_logits = logits[:, -1, :]

        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(last_token_logits, dim=-1)

        max_prob_index = torch.argmax(probabilities, dim=-1)
        logger.info('probabilities_np', max_prob_index)
        
        return max_prob_index, kvCache



class PrefillGPT2Block(nn.Module):
    def __init__(self, config, modelParameters, device, layer_idx=None ):
        super().__init__()
        self.modelParameters = modelParameters
        self.device = device
        self.config = config
        self.embed_dim = self.config.n_embd

        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.layer_idx = layer_idx
        self.attn = PrefillGPT2Attention(config, modelParameters, device=device, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MyGPT2MLP(inner_dim, config, layer_idx, modelParameters, device=device)

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


    def forward(self, hidden_states):
        logger.info("MyGPT2Block forward: ")
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_out, kv = self.attn(hidden_states)
        hidden_states = attn_out + residual


        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states, kv


class PrefillGPT2Attention(nn.Module):
    def __init__(self, config, modelParameters, device, layer_idx=None):
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



    def forward(self, hidden_states):
        logger.info("MyGPT2Attention forward: ")

        batch_size, seq_len, embed_dim = hidden_states.size()
        
        # 生成 Q, K, V
        # hidden_states: [batch_size, seq_len, embed_dim]
        hidden_states_reshaped = hidden_states.view(-1, hidden_states.size(-1))  

        # hidden_states_reshaped: [batch_size * seq_len, embed_dim]
        # self.c_attn:  [embed_dim, 3 * embed_dim]
        # qkv:  [batch_size * seq_len, 3 * embed_dim]
        qkv = self.c_attn(hidden_states_reshaped)  

        # qkv拆分
        # query: [batch_size, seq_len, embed_dim]
        # key: [batch_size, seq_len, embed_dim]
        # value: [batch_size, seq_len, embed_dim]
        query, key, value = torch.split(qkv, self.embed_dim, dim=-1)
        
        # 重塑 Q, K, V 以适应多头
        # query: [batch_size, num_heads, seq_len, head_dim]
        # key: [batch_size, num_heads, seq_len, head_dim]
        # value: [batch_size, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力得分
        # Q @ K^T
        # query: [batch_size, num_heads, seq_len, head_dim]
        # key.transpose(-2, -1): [batch_size, num_heads, head_dim, seq_len]
        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用注意力偏差
        bias = self.attn.bias[:seq_len, :seq_len]
        bias = bias.unsqueeze(0).unsqueeze(0)
        bias = bias.expand(batch_size, self.num_heads, seq_len, seq_len)

        scores = scores + bias

        # 获取下三角掩码
        mask = torch.tril(torch.ones(seq_len, seq_len))

        # 扩展为多个批次
        batch_mask = mask.unsqueeze(0).expand(batch_size, self.num_heads, -1, -1).to(self.device)
        scores = scores.masked_fill(batch_mask == 0, -1e9)
        
        # 计算注意力权重
        # scores: [batch_size, num_heads, seq_len, seq_len]
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = torch.softmax(scores, dim=-1)  
        
        # 应用注意力权重到 V
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        # value: [batch_size, num_heads, seq_len, head_dim]
        # context: [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, value)
        
        # 合并多头
        # context: [batch_size, num_heads, seq_len, head_dim]
        # 转换为 [batch_size, seq_len, num_heads * head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 最终的线性变换
        # context: [batch_size, seq_len, embed_dim]
        context = self.c_proj(context)  

        output = self.resid_dropout(context) 


        return output, (key, value)




class MyGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config, layer_idx, modelParameters, device):
        super().__init__()
        self.embed_dim = config.n_embd
        self.modelParameters = modelParameters
        self.config = config
        self.layer_idx = layer_idx
        self.c_fc = nn.Linear(intermediate_size, self.embed_dim, bias=True)
        self.c_proj = nn.Linear(self.embed_dim, intermediate_size, bias=True)
        ACT2FN = {
            "gelu": nn.functional.gelu,
            "relu": nn.functional.relu,
            "gelu_new": gelu_new,
        }
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

        self.device = device

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
        logger.info("MyGPT2MLP forward: ")

        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    


