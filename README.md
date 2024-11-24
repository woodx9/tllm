## tllm: A Journey to create your own llm inference server

在这个旅程中，您将学习关于LLM推理、KV缓存、静态批处理和连续批处理...的内容，并最终学会如何创建自己的LLM推理服务器。

You will learn about LLM inference, KV cache, static batching, and continuous batching... in this journey, and how to create your own LLM inference server eventually.



### 目录概要  Course Outline
1. gpt2推理
2. LLM采样方法：top-P和top-K 
3. KV cache加速推理 
4. 静态批处理 
5. 连续批处理 
6. Tensor设备并行 
7. PD设备分离 
8. 推测解码

####
1. gpt2 infer
2. LLM sampling methods: Top-p and Top-k
3. KV cache accelerates inference
4. Static batching
5. Continous batching
6. Tensor Device Parallelis
7. PD Device Parallelism
8. Speculative decoding

### 模型下载 Download model
1. 访问 [魔搭社区](https://www.modelscope.cn/models/AI-ModelScope/gpt2) 并下载 pytorch_model.bin 文件。
2. 将下载的文件移动到 model/ 文件夹中。
3. 将文件重命名为 gpt2_pytorch_model.bin。

#### 
1. Go to [Huggging Face](https://huggingface.co/openai-community/gpt2/tree/main) and download the pytorch_model.bin file.
2. Move the downloaded file to the model/ folder.
2. Rename the fileto gpt2_pytorch_model.bin.


### 运行项目示例 Run the project example

```bash
conda create --name tllm python=3.10

conda activate tllm

pip install -r requirement.txt

cd chapter1_gpt2_infer/

python main.py
