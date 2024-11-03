from prefillModel import PrefillModel
from decodeModel import DecodeModel
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from utils import add_new_input_ids


class ContinuousBatchSchedule:
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
        first_batch = prompts[:self.batch_size]

        left_prompts = prompts[self.batch_size:]

        generate_result = self.continuous_batch(first_batch, left_prompts)

        return generate_result
    
    def encodePromptsWihSeqLen(self, prompts , seq_len):
        encodeBatch = self.tokenizer(prompts,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
        )

        

        input_ids = encodeBatch['input_ids']
        
        padding_len = seq_len - input_ids.size(1)


        print('seq_len:', seq_len)
        print('input_ids.size(1):', input_ids.size(1))
        print('padding_len:', padding_len)

        input_id_padding = torch.full((input_ids.size(0), padding_len), self.tokenizer.pad_token_id)
        input_ids = torch.cat((input_ids, input_id_padding), dim=-1)

        attention_mask = encodeBatch['attention_mask']
        attention_mask_padding = torch.full((attention_mask.size(0), padding_len), 0)
        attention_mask = torch.cat((attention_mask, attention_mask_padding), dim=-1)

        print('input_ids:', input_ids.shape)


        return input_ids, attention_mask
    
    # 
    def input_ids_AddPadding(self, input_ids, seq_len):
        padding_len = seq_len - input_ids.size(1)

        input_id_padding = torch.full((input_ids.size(0), padding_len), self.tokenizer.pad_token_id)
        new_input_ids = torch.cat((input_ids, input_id_padding), dim=-1)
        return new_input_ids

    
    def attention_mask_AddPadding(self, attention_mask, seq_len):
        padding_len = seq_len - attention_mask.size(1)

        attention_mask_padding = torch.full((attention_mask.size(0), padding_len), 0)
        attention_mask = torch.cat((attention_mask, attention_mask_padding), dim=-1)

        return attention_mask

    def kv_cache_AddPadding(self, kv, seq_len):
        batch_size, origin_seq_len, embed_dim = kv[0][0].size()
        
        padding_len = seq_len - origin_seq_len

        key_padding = torch.full((batch_size, padding_len, embed_dim), 0)
        value_padding = torch.full((batch_size, padding_len, embed_dim), 0)

        new_kv = [(torch.cat((kv_item[0], key_padding), dim=1), torch.cat((kv_item[1], value_padding), dim=1)) for kv_item in kv]

        print('padding_len: ', padding_len)
        print('kv_cache_AddPadding: ', new_kv[0][0].shape)
        
        return new_kv


    
    def encodePrompts(self, prompts):
        encodeBatch = self.tokenizer(prompts,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
        )

        input_ids = encodeBatch['input_ids']
        attention_mask = encodeBatch['attention_mask']

        return input_ids, attention_mask
    

    def continuous_batch(self, first_batch_prompts, left_prompts):
        input_ids, attention_mask = self.encodePrompts(first_batch_prompts)

        
        new_input_ids, kv = self.prefillModel(input_ids, attention_mask)
        input_ids, attention_mask = add_new_input_ids(self.tokenizer, input_ids, attention_mask, new_input_ids)

        generate_result = []

        # 进行decode
        while input_ids.size(0) > 0:
            new_input_ids, kv = self.decodeModel(input_ids, kv, attention_mask)
            

            input_ids, attention_mask = add_new_input_ids(self.tokenizer, input_ids, attention_mask, new_input_ids)

            seq_len = input_ids.size(1)

            # 拿到所有需要终止的index
            # 1. 生成的token中出现了stop word
            # 2. 生成的token数量超过了max_decode_token_num
            stop_decode_indexs = [
                index 
                for index, (new_id, mask) in enumerate(zip(new_input_ids, attention_mask)) 
                if new_id in self.stop_words_token or mask.sum() > self.max_docode_token_num
            ]

            print('stop decode indexs', stop_decode_indexs)
            print('input_ids', input_ids)

            # 终止decode
            # 存好结果，从input_ids， mask attention，kv cache中删除
            delete_num = 0
            for index in stop_decode_indexs:
                fix_index = index - delete_num
                generate_result.append(input_ids[fix_index])
                input_ids = torch.cat((input_ids[:fix_index], input_ids[fix_index + 1:]), dim=0)
                attention_mask = torch.cat((attention_mask[:fix_index], attention_mask[fix_index + 1:]), dim=0)

                kv = [( torch.cat((kv_item[0][:fix_index], kv_item[0][fix_index + 1:]), dim=0), 
                        torch.cat((kv_item[1][:fix_index], kv_item[1][fix_index + 1:]), dim=0))
                        for kv_item in kv]

                delete_num += 1
                

                
            # 先进行prefill，拿到新的prompt的input_ids和mask attention，kv cache, 然后存进去
            if len(left_prompts) > 0  and len(stop_decode_indexs) > 0:
                need_prefill_number = len(stop_decode_indexs)
                
                print('need prefill number:', need_prefill_number)
                print('seq_len:', seq_len)
                seq_len = 0
                if input_ids.size(0) > 0:
                    print('input_ids when caculate seq_len: ', input_ids)
                    print('mask attention when caculate seq_len: ', attention_mask)

                    sum_over_batches = attention_mask.sum(dim=1)
                    print('sum_over_batches: ', sum_over_batches)
                    seq_len = sum_over_batches.max()


                prefill_input_ids, prefill_attention_mask = self.encodePrompts(left_prompts[:need_prefill_number])

                print('input.ids.size(1)', input_ids.size(1))
                print('input_ids  xxx', input_ids)


                # 这里需要进行裁剪input_ids，attention_mask和kv cache
                if (seq_len  < input_ids.size(1)):
                    print('开始裁减')

                    reduced_sentence_length = input_ids.size(1) - seq_len

                    input_ids = input_ids[:, :-reduced_sentence_length]
                    attention_mask = attention_mask[:, :-reduced_sentence_length]

                    kv = [(kv_item[0][:, :-reduced_sentence_length], kv_item[1][:, :-reduced_sentence_length]) for kv_item in kv]
                    


                

                if (seq_len  > prefill_input_ids.size(1)):
                    print('if happen')
                    
                    prefill_input_ids, prefill_attention_mask = self.encodePromptsWihSeqLen(left_prompts[:need_prefill_number], seq_len - 1)
                    print('prefill_input_ids:', prefill_input_ids.shape)


                    left_prompts = left_prompts[need_prefill_number:]

                    new_prefill_input_ids, new_prefill_kv = self.prefillModel(prefill_input_ids, prefill_attention_mask)

                    new_input_ids, new_attention_mask = add_new_input_ids(self.tokenizer, prefill_input_ids, prefill_attention_mask, new_prefill_input_ids)
                    

                    # 需要考虑input_ids，attention_mask和kv cache的seq_len缩减和扩展
                    print('seq_len:', seq_len)
                    print('input_ids: ', input_ids.shape)
                    print('new_input_ids: ', new_input_ids.shape)

                    input_ids = torch.cat((input_ids, new_input_ids), dim=0)
                    attention_mask = torch.cat((attention_mask, new_attention_mask), dim=0)
                    

                    kv = [(torch.cat((old_kv[0], new_kv[0]), dim=0), 
                        torch.cat((old_kv[1], new_kv[1]), dim=0))
                        for old_kv, new_kv in zip(kv, new_prefill_kv)]
                else :
                    print('else happen')
                    # prefill_input_ids, prefill_attention_mask = self.encodePrompts(left_prompts[:need_prefill_number])
                    # 上面已经执行过了

                    new_prefill_input_ids, new_prefill_kv = self.prefillModel(prefill_input_ids, prefill_attention_mask)

                    new_input_ids, new_attention_mask = add_new_input_ids(self.tokenizer, prefill_input_ids, prefill_attention_mask, new_prefill_input_ids)

                    left_prompts = left_prompts[need_prefill_number:]

                    print('seq_len:', seq_len)
                    print('input_ids: ', input_ids)
                    print('new_input_ids:', new_input_ids)

                    if seq_len != 0:
                        # 这里要考虑input_ids，attention_mask和kv cache的seq_len的增加，长度达到new prefill seq_len的距离
                        new_seq_len = new_input_ids.size(1)
                        input_ids = self.input_ids_AddPadding(input_ids, new_seq_len)
                        attention_mask = self.attention_mask_AddPadding(attention_mask, new_seq_len)
                        # kv cache的seq_len始终比input_ids少一个
                        kv = self.kv_cache_AddPadding(kv, new_seq_len - 1)

                        input_ids = torch.cat((input_ids, new_input_ids), dim=0)
                        attention_mask = torch.cat((attention_mask, new_attention_mask), dim=0)

                        kv = [(torch.cat((old_kv[0], new_kv[0]), dim=0),
                            torch.cat((old_kv[1], new_kv[1]), dim=0))
                            for old_kv, new_kv in zip(kv, new_prefill_kv)]
                    else :
                        input_ids = new_input_ids
                        attention_mask = new_attention_mask
                        kv = new_prefill_kv






        decoded_strings = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generate_result]


        return decoded_strings
