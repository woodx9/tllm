import torch 
import random 
from logger import logger, LogLevel
import numpy as np


class Sampler:
    def __init__(self, temperature: float, top_p: float, top_k: int):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        if not (self.temperature >= 0 and self.temperature <= 1):
            self.temperature = 0
        if not (self.top_p >= 0 and self.top_p <= 1):
            self.top_p = 0.9

    def temperatureSampling(self, logits: torch.Tensor):
        if self.temperature == 0:
            return torch.argmax(logits)
        else:
            adjusted_logits = logits / self.temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)
            return adjusted_probs


    # 只支持batch为1的输入
    def topP(self, logits: torch.Tensor):
        logits = logits.squeeze()
        # 首先使用温度采样
        adjusted_probs = self.temperatureSampling(logits)

        # 然后使用top-p采样
        # 首先对概率进行排序
        sorted_probs, sorted_indices = torch.sort(adjusted_probs, dim=-1, descending=True)
        
        # 计算累积概率, 截断累计p后面的所有概率
        cumulative_prob = torch.cumsum(sorted_probs, dim=-1)

        cutoff_index = torch.where(cumulative_prob > self.top_p)[0][0].item() + 1

        filtered_probs = sorted_probs[:cutoff_index]

        # 剩下的token排序概率，再做一层归一化
        filtered_probs /= filtered_probs.sum()

        filtered_probs = np.array(filtered_probs).flatten()
        
        # 从剩下的token中采样
        chosen_index = random.choices(range(len(filtered_probs)), weights=filtered_probs, k=1)[0]
        # 
        
        return sorted_indices[chosen_index]

    
    def topK(self, logits: torch.Tensor):
        logits = logits.squeeze()

        # 首先使用温度采样
        adjusted_probs = self.temperatureSampling(logits)

        # 然后使用top-p采样
        # 首先对概率进行排序
        sorted_probs, sorted_indices = torch.sort(adjusted_probs, dim=-1, descending=True)

        logger.info('topK sorted probs: ', sorted_probs)

        filtered_probs = sorted_probs[:self.top_k]

        # 剩下的token排序概率，再做一层归一化
        filtered_probs /= filtered_probs.sum()

        filtered_probs = np.array(filtered_probs).flatten()

        chosen_index = random.choices(range(len(filtered_probs)), weights=filtered_probs.tolist(), k=1)[0]
        logger.info('topK filtered probs: ', filtered_probs)

        logger.info('topK chosen index: ', chosen_index)

        return sorted_indices[chosen_index]