from inspect import signature
import torch
from transformers import AutoConfig
from auto_quant.quant.awq.models import *
from typing import Dict, Optional, Union

AWQ_CAUSAL_LM_MODEL_MAP = {
    "mpt": MptAWQForCausalLM,
    "llama": LlamaAWQForCausalLM,
    "opt": OPTAWQForCausalLM,
    "RefinedWeb": FalconAWQForCausalLM,
    "RefinedWebModel": FalconAWQForCausalLM,
    "bloom": BloomAWQForCausalLM,
    "gptj": GPTJAWQForCausalLM
}

def check_and_get_model_type(model_dir, trust_remote_code=True):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type not in AWQ_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type

class AutoAWQForCausalLM:
    def __init__(self):
        raise EnvironmentError('You must instantiate AutoAWQForCausalLM with\n'
                               'AutoAWQForCausalLM.from_quantized or AutoAWQForCausalLM.from_pretrained')
    
    @classmethod
    def from_pretrained(
        self, 
        model_path, 
        quant_config: AWQConfig, 
        max_memory: Optional[dict] = None,
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code=True,
        **kwargs
    ) -> BaseAWQForCausalLM:
        
        model_type = check_and_get_model_type(model_path, trust_remote_code)
        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
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
        trust_remote_code=True,
        **kwargs
    ) -> BaseAWQForCausalLM:
        model_type = check_and_get_model_type(quant_path, trust_remote_code)
        quant_func = AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized
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
