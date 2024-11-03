from transformers import GPT2Tokenizer
import time
import matplotlib.pyplot as plt
from myGPT2Model import MyGPT2Model
import numpy as np
import torch

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode("I like to ", return_tensors="pt")

    print(input_ids)
    myModel = MyGPT2Model()
    infer_start_time = time.time()
    infer_per_token_time = []
    new_token_number = 100


    with torch.no_grad():
        for i in range(new_token_number):
            infer_token_start_time = time.time()
            result = myModel(input_ids)
            infer_per_token_time.append((time.time() - infer_token_start_time))
            print('获得新的token id: ', result)
            new_id = torch.tensor([[result]])
            input_ids = torch.cat((input_ids, new_id), dim=1)
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