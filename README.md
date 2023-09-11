# AutoQuant
**I'm currently working on adding GPTQ.**

An easy-to-use LLMs quantization package

Unlike [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) and [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), AutoQuant aims to be a simple but expandable package. So if you want speed, I don't recommend using AutoQuant.

## Install
### Build source
```
git clone https://github.com/qwopqwop200/AutoQuant
cd AutoQuant
pip install -e .
```
## Usage
Below, you will find examples of how to easily quantize a model and run inference.
### Quantization

```python
import torch
from transformers import AutoTokenizer
from auto_quant import AutoAWQForCausalLM, AWQConfig
from datasets import load_dataset

def get_calib_dataset(tokenizer, n_samples=512, block_size=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

pretrained_model_dir = "facebook/opt-125m"
quant_model_dir = "opt-125m-awq"

quant_config = AWQConfig(bits=4,group_size=128)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
model = AutoAWQForCausalLM.from_pretrained(pretrained_model_dir, quant_config)

model.quantize(get_calib_dataset(tokenizer))

model.save_quantized(quant_model_dir,max_shard_size='100MB')
tokenizer.save_pretrained(quant_model_dir)
```

### Inference

Run inference on a quantized model from Huggingface:

```python
from auto_quant import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = "casperhansen/vicuna-7b-v1.5-awq"

model = AutoAWQForCausalLM.from_quantized(quant_path)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

model.generate(...)
```

## Model Quantization
1. [GPTQ](https://arxiv.org/abs/2210.17323) released with the paper [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) by Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
2. [AWQ](https://arxiv.org/abs/2306.00978) released with the paper [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) by Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Song Han

## Acknowledgement
This quantization package is inspired by the following projects: [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ),[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
