from prefillModel import PrefillModel
from decodeModel import DecodeModel
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from utils import add_new_input_ids


class StaticBatchSchedule:
    def __init__(self, batch_size, max_context_token_num, stop_words):
        self.batch_size = batch_size
        self.prefillModel = PrefillModel()
        self.decodeModel = DecodeModel()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # 每个批次，放入的最多的prompt数量
        self.batch_size = batch_size
        # 产生的最大token数量
        self.max_docode_token_num = max_context_token_num
        self.stop_words_token =  input_ids = self.tokenizer.convert_tokens_to_ids(stop_words)
    

    def process(self, prompts):
        # 对prompts进行分组
        batches = [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]


        result = []

        for batch in batches:
            print('batch', batch)
            batch_result = self.process_batch(batch)
            result.extend(batch_result)

        # 过滤掉所有的stop word
        print('stop words', self.stop_words_token)

        return result
    

    def process_batch(self, prompts):
        encodeBatch = self.tokenizer(prompts,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
        )

        input_ids = encodeBatch['input_ids']
        attention_mask = encodeBatch['attention_mask']
        print('input_ids:', input_ids)
        print('attention_mask', attention_mask)

        
        new_input_ids, kv = self.prefillModel(input_ids, attention_mask)
        input_ids, attention_mask = add_new_input_ids(self.tokenizer, input_ids, attention_mask, new_input_ids)

        # 进行decode
        for i in range(self.max_docode_token_num ):
            new_input_ids, kv = self.decodeModel(input_ids, kv, attention_mask)
            input_ids, attention_mask = add_new_input_ids(self.tokenizer, input_ids, attention_mask, new_input_ids)

            # 如果所有句子都有stop word就停止decode
            stop_decode_num = 0
            for str in input_ids:
                filtered_str = []
                for token_id in str:
                    if token_id in self.stop_words_token:
                        stop_decode_num += 1
                        break
            if (stop_decode_num == self.batch_size):
                print('stop decode')
                break

        # 过滤掉所有的stop word之后的内容
        filtered_input_ids = []
        for str in input_ids:
            filtered_str = []
            for token_id in str:
                if token_id not in self.stop_words_token:
                    filtered_str.append(token_id)
                else:
                    break
            filtered_input_ids.append(filtered_str)

        input_ids = filtered_input_ids

        decoded_strings = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]


        return decoded_strings












    
    
    
