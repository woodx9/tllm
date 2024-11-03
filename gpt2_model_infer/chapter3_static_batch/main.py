from transformers import GPT2Tokenizer
import time
import matplotlib.pyplot as plt
from prefillModel import PrefillModel
from decodeModel import DecodeModel
import numpy as np
import torch
from static_batch_schedule import StaticBatchSchedule

def main():
    staticBatchSchedule =  StaticBatchSchedule(2, 20, ['.', ','])

    prompts = ["this book is ", "it's too late to", 'love is cure', "apple is good", "nice to", "my name is",]

    res = staticBatchSchedule.process(prompts)
    print("获得结果")
    for (index, str) in enumerate(res):
        print(index, "---------------")
        print(str)



main()