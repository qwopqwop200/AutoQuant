import os
import gc
import logging
import copy
from tqdm import tqdm
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence

from huggingface_hub import snapshot_download
from transformers.modeling_utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_in_model, cpu_offload_with_hook
from accelerate.utils import get_balanced_memory
from accelerate.hooks import remove_hook_from_module

from auto_quant.models.base import BaseQuantConfig, BaseQuantForCausalLM
from auto_quant.modules.qlinear_exllama import ExllamaLinear, exllama_post_init, make_sure_no_tensor_in_meta_device, check_exllama_can_save
from auto_quant.utils.module import get_device, move_to_device, get_named_module, get_module_by_name_prefix, get_module_by_name_suffix, set_op_by_name, simple_dispatch_model
from ..quantize import RTN

@dataclass
class RTNConfig(BaseQuantConfig):
    bits: int = field(default=4, metadata={"choices": [4]})
    sym: bool = field(default=False)
    quant_type = 'RTN'
    
    def __post_init__(self):
        fields_info = fields(self)

        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(f"only support quantize to {fields_info[0].metadata['choices']} bits.")

    def to_dict(self):
        return {
            "bits": self.bits,
            "sym": self.sym,
            "quant_type": self.quant_type,
        }


class BaseRTNForCausalLM(BaseQuantForCausalLM):
    layer_type: str = None
    layers_block_name: str = None
    outside_layer_modules: List[str] = None
    inside_layer_modules: List[List[str]] = None
    lm_head_name: str = "lm_head"
    def __init__(self, model, quant_config, is_quantized):
        super().__init__(model, quant_config, is_quantized)
        self.quant_config:RTNConfig = quant_config
        self.search_result = None
                
    @classmethod
    def check_quant_config(self, quant_config: RTNConfig):
        pass

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
        **kwargs,
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        snapshot_cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }
    
        if not os.path.isdir(quant_path):
            quant_path = snapshot_download(quant_path,ignore_patterns=[ "*.msgpack" , "*.h5"], **snapshot_cached_file_kwargs)

        config = AutoConfig.from_pretrained(quant_path, trust_remote_code=trust_remote_code, **cached_file_kwargs)
        quant_config = RTNConfig.from_pretrained(quant_path, **cached_file_kwargs)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
        
        self._load_quantized_modules(self, model, quant_config)
        
        model.tie_weights()
        
        if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )
        if isinstance(device_map, dict):
            max_memory = None
        else:
            if device is None and not device_map and not max_memory:
                device_map = "auto"
            if device is not None:
                device = torch.device(device)
                if not max_memory and not device_map:
                    device_map = {"": device.index if device.type == "cuda" else device.type}
            if not isinstance(device_map, dict) and device_map != "sequential":
                max_memory = get_balanced_memory(
                    model=model,
                    max_memory=max_memory,
                    no_split_module_classes=[self.layer_type],
                    low_zero=(device_map == "balanced_low_0")
                )
        if not isinstance(device_map, dict):
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[self.layer_type]
            )

        if low_cpu_mem_usage:
            make_sure_no_tensor_in_meta_device(model)
        
        if os.path.isfile(os.path.join(quant_path, WEIGHTS_NAME)):
            quant_path = os.path.join(quant_path, WEIGHTS_NAME)
        elif os.path.isfile(os.path.join(quant_path, SAFE_WEIGHTS_NAME)):
            quant_path = os.path.join(quant_path, SAFE_WEIGHTS_NAME)
        load_checkpoint_in_model(
            model,
            checkpoint=quant_path,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True
        )
        model = dispatch_model(model, device_map)
        model = exllama_post_init(model, False)
        return self(model, quant_config, is_quantized=True)
        
    def _load_quantized_modules(self, model, quant_config):
        layers = get_named_module(model)
        ignore_layers = [self.lm_head_name] + self.outside_layer_modules
        for name in list(layers.keys()):
            if any([name.startswith(ignore_layer) for ignore_layer in ignore_layers]):
                del layers[name]

        for name in tqdm(layers.keys(), desc="Replacing layers..."):
            layer = layers[name]
            q_linear = ExllamaLinear.from_linear(
                    layer, 
                    quant_config.bits, 
                    -1,
                    False,
                    True)
            q_linear.to(get_device(layer))
            set_op_by_name(model, name, q_linear)
            torch.cuda.empty_cache()
            gc.collect()
        
    @staticmethod
    def _resize_attention_mask(attention_mask: List[torch.LongTensor]):
        return attention_mask

    @staticmethod
    def _resize_position_ids(position_ids: List[torch.LongTensor]):
        return position_ids

    @torch.inference_mode()
    def quantize(self):
        device_map = self.hf_device_map
        layers = get_module_by_name_prefix(self.model, self.layers_block_name)
        inside_layer_modules = self.inside_layer_modules
        quantizers = {}
        for i in tqdm(range(len(layers)), desc="RTN Quantization"):
            layer = layers[i]
            force_layer_back_to_cpu = False
            if get_device(layer) == 'cpu':
                move_to_device(layer, 'cuda:0')
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            full = get_named_module(layer)
            for names in inside_layer_modules:
                subset = {n: full[n] for n in names}
                rtn = {}
                for name in subset:
                    rtn[name] = RTN(subset[name])
                    rtn[name].quantizer.configure(
                        self.quant_config.bits,
                        perchannel=True,
                        sym=self.quant_config.sym,
                        mse=False,
                    )
                    
                for name in subset:
                    scale, zero, g_idx = rtn[name].fasterquant()
                    quantizers[f'{self.layers_block_name}.{i}.{name}'] = (
                        move_to_device(scale, 'cpu' if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(zero, 'cpu' if force_layer_back_to_cpu else cur_layer_device),
                        g_idx
                    )

            layers[i] = move_to_device(layer, 'cpu' if force_layer_back_to_cpu else cur_layer_device)
            del layer
            del rtn
            torch.cuda.empty_cache()

        self.pack_model(quantizers=quantizers)
        if device_map:
            self.model = simple_dispatch_model(self.model, device_map)
        logging.warning("Exllama has not been post-initialized. inference doesn't work. Please save the model and then load it.")
        self.is_quantized = True
        torch.cuda.empty_cache()
            
    def pack_model(self, quantizers):
        layers = get_named_module(self.model)
        layers = {n: layers[n] for n in quantizers}
        for name in tqdm(layers, desc="Pack Model"):
            scale, zero, g_idx = quantizers[name]
            layer_device = get_device(layers[name])
            layer, scale, zero, g_idx = layers[name].to('cpu'), scale.to('cpu'), zero.to('cpu'), g_idx
            qlayer = ExllamaLinear.from_linear(
                    layer, 
                    self.quant_config.bits, 
                    -1,
                    False,
                    False, 
                    scale, 
                    zero,
                    g_idx)
            qlayer.to(layer_device)
            set_op_by_name(self.model, name, qlayer)
            torch.cuda.empty_cache()
            gc.collect()