### This is tllm Draft

### download gpt2 model

[huggging face link](https://huggingface.co/openai-community/gpt2/tree/main)

download pytorch_model.bin, put it under gpt2_model_infer/model/, and rename it to "gpt2_pytorch_model.bin"

### how to run 
```javascript
conda create --name tllm python=3.10

conda activate tllm

pip install -r requirement.txt

cd gpt2_model_infer/chapter1_gpt2_infer/

python main.py
