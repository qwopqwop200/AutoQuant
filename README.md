# AutoQuant
An easy-to-use LLMs quantization package

Unlike [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) and [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), AutoQuant aims to be a simple but expandable package. So if you want speed, I don't recommend using AutoQuant.

*Latest News* 🔥
- [2023/09] Support GPTQ Quantization method

## Install
### Build source
```
pip install git+https://github.com/qwopqwop200/AutoQuant
```
## Usage
Below, you will find examples of how to easily quantize a model and run inference.
### Quantization

```python
import torch
from transformers import AutoTokenizer
from auto_quant import AutoQuantForCausalLM, AutoQuantConfig, get_calib_dataset
from datasets import load_dataset

pretrained_model_dir = "facebook/opt-125m"
quant_model_dir = "opt-125m-awq"

quant_config = AutoQuantConfig('AWQ',bits=4,group_size=128)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
model = AutoQuantForCausalLM.from_pretrained(pretrained_model_dir, quant_config)

model.quantize(get_calib_dataset(tokenizer, 'pile'))

model.save_quantized(quant_model_dir ,use_safetensors=True)
tokenizer.save_pretrained(quant_model_dir)
```

### Inference

```python
from auto_quant import AutoQuantForCausalLM
from transformers import AutoTokenizer

quant_path = "opt-125m-awq"

model = AutoQuantForCausalLM.from_quantized(quant_path, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

print(tokenizer.decode(model.generate(**tokenizer("auto_quant is", return_tensors="pt").to(model.device))[0]))
```

### Evaluation
```python
from auto_quant import AutoQuantForCausalLM, LMEvalAdaptor
from transformers import AutoTokenizer
from lm_eval import evaluator

quant_path = "opt-125m-awq"

model = AutoQuantForCausalLM.from_quantized(quant_path)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

lm_eval_model = LMEvalAdaptor(model, tokenizer, batch_size=1)

results = evaluator.simple_evaluate(model=lm_eval_model,tasks=['wikitext'],batch_size=1,no_cache=True,num_fewshot=0,)
print(evaluator.make_table(results))
```

## Model Quantization
1. [GPTQ](https://arxiv.org/abs/2210.17323) released with the paper [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) by Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
2. [AWQ](https://arxiv.org/abs/2306.00978) released with the paper [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) by Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Song Han

## Acknowledgement
This quantization package is inspired by the following projects: [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ),[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
