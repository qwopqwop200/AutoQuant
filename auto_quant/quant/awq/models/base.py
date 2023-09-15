import os
import gc
import functools
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from accelerate.utils import get_balanced_memory

from auto_quant.models.base import BaseQuantConfig, BaseQuantForCausalLM
from auto_quant.modules.qlinear_exllama import ExllamaLinear, exllama_post_init, make_sure_no_tensor_in_meta_device
from auto_quant.utils.module import append_str_prefix, get_op_name, get_named_module, set_op_by_name, simple_dispatch_model, get_device
from ..quantize.quantizer import pseudo_quantize_tensor, ScaledActivation
from ..quantize.auto_clip import auto_clip_block, apply_clip
from ..quantize.auto_scale import auto_scale_block, apply_scale
@dataclass
class AWQConfig(BaseQuantConfig):
    bits: int = field(default=4, metadata={"choices": [4]})
    group_size: int = field(default=-1)
    zero_point: bool = field(default=True)
    auto_scale: bool = field(default=True)
    mse_range: bool = field(default=True)
    inplace: bool = field(default=False)
    quant_type = 'AWQ'
    
    def __post_init__(self):
        fields_info = fields(self)

        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(f"only support quantize to {fields_info[0].metadata['choices']} bits.")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("unless equal to -1, group_size must greater then 0.")
        if not self.zero_point:
            raise NotImplementedError("We only support zero_point quantization now.")
                
    def to_dict(self):
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "zero_point": self.zero_point,
            "auto_scale": self.auto_scale,
            "mse_range": self.mse_range,
            "inplace": self.inplace,
            "quant_type": self.quant_type,
        }

class BaseAWQForCausalLM(BaseQuantForCausalLM):
    def __init__(self, model, quant_config, is_quantized):
        super().__init__(model, quant_config, is_quantized)
        self.quant_config:AWQConfig = quant_config
        self.search_result = None
                
    @classmethod
    def check_quant_config(self, quant_config: AWQConfig):
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
        quant_config = AWQConfig.from_pretrained(quant_path, **cached_file_kwargs)

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
        model = simple_dispatch_model(model, device_map)
        model = exllama_post_init(model, False)
        return self(model, quant_config, is_quantized=True)

    def _load_quantized_modules(self, model, quant_config):
        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_module(layer)

            # Replace activation functions
            self._scale_activations(self, layer)

            # Replace nn.Linear with ExllamaLinear
            for name, module in named_linears.items():
                q_linear = ExllamaLinear.from_linear(module, quant_config.bits, quant_config.group_size, False, True)
                q_linear.to(get_device(layer))
                set_op_by_name(layer, name, q_linear)
            torch.cuda.empty_cache()
            gc.collect()
            
    def _prepare_examples_for_quantization(self, examples: List[Union[List[int], torch.LongTensor]]):
        pad_token_id = self.model.config.pad_token_id
        if not pad_token_id:
            pad_token_id = self.config.eos_token_id
    
        if isinstance(examples,list):
            new_examples = []
            for tensor in examples:
                if tensor.dim() > 2:
                    raise Exception('examples must be 2D tensor or less.')
                else:
                    for i in range(tensor.dim(),2):
                        tensor.squeeze(0)
                        
                for i in range(tensor.shape[0]):
                    new_examples.append(tensor[i])
            examples = pad_sequence(new_examples, True, padding_value=pad_token_id)
            
        if examples.dim() > 2:
            raise Exception('examples must be 2D tensor or less.')
        else:
            for i in range(examples.dim(),2):
                examples.squeeze(0)
        return examples
            
    @torch.no_grad()
    def quantize(self, examples: List[Union[List[int], torch.LongTensor]]):
        examples = self._prepare_examples_for_quantization(examples)
        self._awq_search(examples)
        self._awq_quant()
        self.model = exllama_post_init(self.model, False)
        self.is_quantized = True
        
    def _awq_search(self, examples):
        device_map = self.hf_device_map
        layers = self.get_model_layers(self.model)
        inps = []
        layer_kwargs = {}

        layers[0] = layers[0].cuda()
        self.move_embed(self.model, "cuda")
        
        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, hijacked_inputs, **kwargs):
                inps.append(hijacked_inputs)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        layers[0] = Catcher(layers[0])
        try:
            self.model(examples.to(get_device(self.model)))
        except ValueError:  # work with early exit
            pass
        del examples
        layers[0] = layers[0].module  # restore
        inps = inps[0]

        layers[0] = layers[0].cpu()
        self.move_embed(self.model, "cpu")
        
        gc.collect()
        torch.cuda.empty_cache()
        awq_results = {"scale": [],"clip": [],}

        # Run AWQ search layer by layer
        for i in tqdm(range(len(layers)), desc="AWQ Search"):
            layer = layers[i].cuda()
            named_linears = get_named_module(layer)

            # firstly, get input features of all linear layers
            def cache_input_hook(m, x, y, name, feat_dict):
                x = x[0]
                x = x.detach().cpu()
                feat_dict[name].append(x)

            input_feat = defaultdict(list)
            handles = []
            for name in named_linears:
                handles.append(named_linears[name].register_forward_hook(functools.partial(cache_input_hook, name=name,feat_dict=input_feat)))
            inps = inps.to(get_device(layer))  # in case multi-gpu
            # get output as next layer's input
            inps = layer(inps, **layer_kwargs)[0]
            for h in handles:
                h.remove()
            # now solve for scaling and clipping
            input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

            # Clear GPU memory
            torch.cuda.empty_cache()

            if self.quant_config.auto_scale:  # if it applies, we should also modify the input_feat with scales
                scales_list = auto_scale_block(
                    self,
                    layer,
                    layer_kwargs,
                    quant_config=self.quant_config,
                    input_feat=input_feat,
                )

                apply_scale(layers[i], scales_list, input_feat_dict=input_feat)

                # append prefix to make names global
                awq_results["scale"] += append_str_prefix(scales_list, get_op_name(self.model, layer) + ".")

            # Clear GPU memory
            torch.cuda.empty_cache()
            
            if self.quant_config.mse_range:
                clip_list = auto_clip_block(layer,quant_config=self.quant_config,input_feat=input_feat)
                apply_clip(layer, clip_list)
                # append prefix to make names global
                awq_results["clip"] += append_str_prefix(clip_list, get_op_name(self.model, layer) + ".")

            layer = layer.cpu()
            # Haotian: check activation replacement
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()
        
        self.search_result = awq_results
        if device_map:
            self.model = simple_dispatch_model(self.model, device_map)

    def _awq_quant(self):
        layers = self.get_model_layers(self.model)

        # Run AWQ quantization
        for i in tqdm(range(len(layers)), desc="AWQ Quantization"):
            layer = layers[i]
            named_linears = get_named_module(layer)
            self._scale_activations(self, layer)

            for name, module in named_linears.items():
                device = get_device(module)
                module.cuda()
                
                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, 
                    self.quant_config.bits, 
                    self.quant_config.group_size,
                    self.quant_config.zero_point, 
                    self.quant_config.inplace,
                    get_scale_zp=True, 
                )
                q_linear = ExllamaLinear.from_linear(
                    module, 
                    self.quant_config.bits, 
                    self.quant_config.group_size,
                    False,
                    False, 
                    scales, 
                    zeros
                )

                module.cpu()
                q_linear.to(device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()
            
    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer)
        if scale_dict['is_scalable']:
            if not isinstance(scale_dict['scale_layer'], ScaledActivation):
                device = get_device(layer)
                
                # get activation scale
                scale_like = torch.ones(scale_dict['scale_shape'], dtype=param.dtype, device=device)
                
                # scale activation
                scaled_act = ScaledActivation(scale_dict['scale_layer'], scale_like)
                set_op_by_name(layer, scale_dict['scale_name'], scaled_act)