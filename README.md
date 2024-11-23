## This is tllm Draft: A Journey to set up your own llm server

you will learn about llm infer, kv cache, static batch and continuous batch... in this journery

### download gpt2 model

[huggging face link](https://huggingface.co/openai-community/gpt2/tree/main)

download pytorch_model.bin, put it under model/, and rename it to "gpt2_pytorch_model.bin"

### how to run 
```javascript
conda create --name tllm python=3.10

conda activate tllm

pip install -r requirement.txt

cd chapter1_gpt2_infer/

python main.py
