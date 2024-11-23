import torch
import numpy as np


def getLastPosition(attention_mask: torch.Tensor):
    # 反转 attention_mask 每行的顺序
    reversed_attention_mask = attention_mask.flip(dims=[1])

    # 找到每行第一个 1 的位置，即原始张量中每行 1 的最后位置
    last_positions = reversed_attention_mask.argmax(dim=1).tolist()

    # 由于是反转序列，所以需要用 seq_length 减去得到的位置减一
    seq_length = attention_mask.size(1)

    #  获取每个prompt的最后一个token位置, 因此来找到计算概率的隐向量
    last_positions = torch.tensor([seq_length - pos - 1 for pos in last_positions])

    return last_positions

def add_new_input_ids(tokenizer, input_ids, attention_mask, new_input_ids, device):
    new_input_id_column = torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, device=device)
    new_attn_mask_column = torch.full((attention_mask.size(0), 1), 0, device=device)

    # 找到每行第一个 0 的位置，即生成新token的位置
    input_ids = torch.cat((input_ids, new_input_id_column), dim=1)
    attention_mask = torch.cat((attention_mask, new_attn_mask_column), dim=1)

    cpu_attention_mask = attention_mask.to('cpu')
    last_positions = np.argmax(cpu_attention_mask == 0, axis=1)
    for index, element in enumerate(last_positions):
        input_ids[index, element] = new_input_ids[index]
        attention_mask[index, element] = 1

    return input_ids, attention_mask

def get_device():
    device =  'cpu'
    # 这里没有考虑多卡多设备的情况
    if torch.cuda.is_available():
        device = 'cuda'
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'

    return device