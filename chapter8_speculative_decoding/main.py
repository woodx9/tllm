from transformers import GPT2Tokenizer
import time
import matplotlib.pyplot as plt
from prefillModel import PrefillModel
import numpy as np
import torch


# source from chapter6

from utils import get_device

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    original_text = "I am a big fan of NBA, I watch"
    speculative_text = " it every day."

    original_input_ids = tokenizer.encode(original_text, return_tensors="pt")
    print('原始文本的token id: ', original_input_ids)
    speculative_text_ids = tokenizer.encode(speculative_text, return_tensors="pt")
    print('预测文本的token id: ', speculative_text_ids)
    input_ids = tokenizer.encode(original_text + speculative_text, return_tensors="pt")
    print('合并后的token id: ', input_ids)


    print(input_ids)

    device = get_device()
    

    myModel = PrefillModel(device1=device, device2=device)
    infer_start_time = time.time()
    infer_per_token_time = []
    # 只能执行一次，后续要把speculative_text_ids置空
    new_token_number = 1


    with torch.no_grad():
        for i in range(new_token_number):
            infer_token_start_time = time.time()
            input_ids, kvCache = myModel(input_ids, speculative_text_ids)
            speculative_text = ""
            speculative_text_ids = tokenizer.encode(speculative_text, return_tensors="pt")
            
            infer_per_token_time.append((time.time() - infer_token_start_time))

            decoded_string = tokenizer.decode(input_ids[0], skip_special_tokens=True)

            print('当前生成的最新字符串:\n*********\n', decoded_string, '\n***********\n')



    infer_time = time.time() - infer_start_time
    # print('产生每个新token所需的时间: ', infer_per_token_time) 
    print('本次推理的时间总: ', infer_time)
    print('token产生需要的平均时间: ', infer_time / new_token_number)
   
    # 创建一个数组表示x轴的索引
    x = np.arange(len(infer_per_token_time))

    # 创建绘图
    plt.plot(x, infer_per_token_time)
    # plt.show()
            

main()