from transformers import GPT2Tokenizer
import time
import matplotlib.pyplot as plt
from prefillModel import PrefillModel
from decodeModel import DecodeModel
import numpy as np
import torch
from continuous_batch_schedule import ContinuousBatchSchedule

def main():
    continuousBatchSchedule =  ContinuousBatchSchedule(3, 20, ['.', ','])

    prompts1 = ["this book is ", "it's too late to", 'love is cure', "apple is good", "nice to", "my name is",]
    prompts2 = ["this is a long story", "this book is ", "I still love", "it's too late to", 'love is cure', "apple is good", "nice to", "my name is", "this guy is a very bad guy", "my dog is"]

    res = continuousBatchSchedule.process(prompts2)
    print("获得结果")
    for (index, str) in enumerate(res):
        print(index, "---------------")
        print(str)



main()