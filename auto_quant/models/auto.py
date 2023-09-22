import os
import json
from inspect import signature
from typing import Dict, Optional, Union

import torch
from transformers import AutoConfig
from transformers.utils.hub import cached_file
from .base import BaseQuantConfig, BaseQuantForCausalLM
from auto_quant.quant.rtn.models import AutoRTNForCausalLM, RTNConfig
from auto_quant.quant.awq.models import AutoAWQForCausalLM, AWQConfig
from auto_quant.quant.gptq.models import AutoGPTQForCausalLM, GPTQConfig
from auto_quant.quant.quip.models import AutoQuIPForCausalLM, QuIPConfig

QUANT_CONFIG_MAP = {
    "RTN": RTNConfig,
    "AWQ": AWQConfig,
    "GPTQ": GPTQConfig,
    "QuIP": QuIPConfig,
}

QUANT_CAUSAL_LM_MODEL_MAP = {
    "RTN": AutoRTNForCausalLM,
    "AWQ": AutoAWQForCausalLM,
    "GPTQ": AutoGPTQForCausalLM,
    "QuIP": AutoQuIPForCausalLM,
}

def check_quant_type(quant_type):
    if quant_type not in QUANT_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{quant_type} isn't supported yet. Only support: {list(QUANT_CAUSAL_LM_MODEL_MAP.keys())}")
    return quant_type
    
class AutoQuantConfig:
    def __init__(self, quant_type: str, **kwargs):
        quant_type = check_quant_type(quant_type)
        self.quant_config =  QUANT_CONFIG_MAP[quant_type](**kwargs)

    @classmethod
    def from_pretrained(self, save_dir: str, **kwargs):
        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        commit_hash = kwargs.pop("_commit_hash", None)

        quant_config_filename = "quant_config.json"
        if os.path.isdir(save_dir):  # Local
            resolved_config_file = os.path.join(save_dir, quant_config_filename)
        else: # Remote
            resolved_config_file = cached_file(
                save_dir,
                quant_config_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                use_auth_token=use_auth_token,
                revision=revision,
                local_files_only=local_files_only,
                subfolder=subfolder,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                _commit_hash=commit_hash,
            )
        with open(resolved_config_file, "r", encoding="utf-8") as f:
            quant_type = json.loads(f.read())['quant_type']
        quant_type = check_quant_type(quant_type)
        return QUANT_CONFIG_MAP[quant_type].from_pretrained(save_dir, **kwargs)
        
    def __repr__(self):
        return self.quant_config.__repr__()
            
    def __str__(self):
        return self.quant_config.__str__()
        
class AutoQuantForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "AutoQuantForCausalLM is designed to be instantiated\n"
            "using `AutoQuantForCausalLM.from_pretrained` if want to quantize a pretrained model.\n"
            "using `AutoQuantForCausalLM.from_quantized` if want to inference with quantized model."
        )

    @classmethod
    def from_pretrained(
        self, 
        model_path, 
        quant_config, 
        max_memory: Optional[dict] = None,
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = False,
        **kwargs
    ) -> BaseQuantForCausalLM:
        if isinstance(quant_config, AutoQuantConfig):
            quant_config = quant_config.quant_config
        quant_type = check_quant_type(quant_config.quant_type)
        return QUANT_CAUSAL_LM_MODEL_MAP[quant_type].from_pretrained(
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
    ) -> BaseQuantForCausalLM:
        quant_type = AutoQuantConfig.from_pretrained(quant_path).quant_type
        quant_func = QUANT_CAUSAL_LM_MODEL_MAP[quant_type].from_quantized
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