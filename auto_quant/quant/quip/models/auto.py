from inspect import signature
from typing import Dict, Optional, Union

import torch
from transformers import AutoConfig
from auto_quant.quant.quip.models import *

QuIP_CAUSAL_LM_MODEL_MAP = {
    "bloom": BloomQuIPForCausalLM,
    "gpt_neox": GPTNeoXQuIPForCausalLM,
    "gptj": GPTJQuIPForCausalLM,
    "gpt2": GPT2QuIPForCausalLM,
    "llama": LlamaQuIPForCausalLM,
    "opt": OPTQuIPForCausalLM,
    "moss": MOSSQuIPForCausalLM,
    "gpt_bigcode": GPTBigCodeQuIPForCausalLM,
    "codegen": CodeGenQuIPForCausalLM,
    "RefinedWebModel": RWQuIPForCausalLM,
    "RefinedWeb": RWQuIPForCausalLM,
    "falcon": RWQuIPForCausalLM,
    "baichuan": BaiChuanQuIPForCausalLM,
    "internlm": InternLMQuIPForCausalLM,
    "qwen": QwenQuIPForCausalLM,
    "mpt": MptQuIPForCausalLM,
}

def check_and_get_model_type(model_dir, trust_remote_code=True):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type not in QuIP_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet. Only support: {list(QuIP_CAUSAL_LM_MODEL_MAP.keys())}")
    model_type = config.model_type
    return model_type

class AutoQuIPForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "AutoQuIPModelForCausalLM is designed to be instantiated\n"
            "using `AutoQuIPModelForCausalLM.from_pretrained` if want to quantize a pretrained model.\n"
            "using `AutoQuIPModelForCausalLM.from_quantized` if want to inference with quantized model."
        )

    @classmethod
    def from_pretrained(
        self, 
        model_path, 
        quant_config: QuIPConfig, 
        max_memory: Optional[dict] = None,
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = False,
        **kwargs
    ) -> BaseQuIPForCausalLM:
        
        model_type = check_and_get_model_type(model_path, trust_remote_code)
        return QuIP_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path, 
            quant_config=quant_config,
            max_memory=max_memory,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs
        )

    @classmethod
    def from_quantized(
        self, 
        quant_path, 
        device: Optional[Union[str, int]] = None,
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        max_memory: Optional[dict] = None,
        low_cpu_mem_usage: bool = False,
        torch_dtype: torch.dtype = torch.float16, 
        trust_remote_code: bool = False,
        **kwargs
    ) -> BaseQuIPForCausalLM:
        model_type = check_and_get_model_type(quant_path, trust_remote_code)
        quant_func = QuIP_CAUSAL_LM_MODEL_MAP[model_type].from_quantized
        # A static list of kwargs needed for huggingface_hub
        huggingface_kwargs = [
            "cache_dir",
            "force_download",
            "proxies",
            "resume_download",
            "local_files_only",
            "use_auth_token",
            "revision",
            "subfolder",
            "_raise_exceptions_for_missing_entries",
            "_commit_hash"
        ]
        keywords = {
            key: kwargs[key]
            for key in list(signature(quant_func).parameters.keys()) + huggingface_kwargs
            if key in kwargs
        }
        return quant_func(
            quant_path,
            device=device,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype, 
            trust_remote_code=trust_remote_code, 
            **kwargs
        )