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
from ..quantize import QuIP

@dataclass
class QuIPConfig(BaseQuantConfig):
    bits: int = field(default=4, metadata={"choices": [4]})
    quant: str = field(default='ldlbal_admm', metadata={"choices": ['allbal', 'ldlq', 'ldlqRG', 'ldlbal_admm']})
    damp_percent: float = field(default=0.01)
    true_sequential: bool = field(default=True)
    npasses: int = field(default=0)
    pre_gptqH: bool = field(default=False)
    pre_rescale: bool = field(default=False)
    pre_proj: bool = field(default=False)
    pre_proj_exta: bool = field(default=0, metadata={"choices": [0,1,2]})
    qfn: str = field(default='a', metadata={"choices": ['a', 'b', 'c']})
    unbiased: bool = field(default=False)
    lazy_batch: bool = field(default=False)
    quant_type = 'QuIP'
    
    def __post_init__(self):
        fields_info = fields(self)
        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(f"only support quantize to {fields_info[0].metadata['choices']} bits.")
        if self.quant not in fields_info[1].metadata["choices"]:
            raise ValueError(f"'quant' accepts only the following values: {fields_info[1].metadata['choices']}.")      
        if self.pre_proj_exta not in fields_info[8].metadata["choices"]:
            raise ValueError(f"'pre_proj_exta' accepts only the following values: {fields_info[9].metadata['choices']}.")
        if self.qfn not in fields_info[9].metadata["choices"]:
            raise ValueError(f"'qfn' accepts only the following values{fields_info[10].metadata['choices']}.")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def to_dict(self):
        return {
            "bits": self.bits,
            "quant": self.quant,
            "damp_percent": self.damp_percent,
            "true_sequential": self.true_sequential,
            "npasses": self.npasses,
            "pre_gptqH": self.pre_gptqH,
            "pre_rescale": self.pre_rescale,
            "pre_proj": self.pre_proj,
            "pre_proj_exta": self.pre_proj_exta,
            "qfn": self.qfn,
            "unbiased": self.unbiased,
            "lazy_batch": self.lazy_batch,
            "quant_type": self.quant_type,
        }


class BaseQuIPForCausalLM(BaseQuantForCausalLM):
    layer_type: str = None
    layers_block_name: str = None
    outside_layer_modules: List[str] = None
    inside_layer_modules: List[List[str]] = None
    lm_head_name: str = "lm_head"
    def __init__(self, model, quant_config, is_quantized):
        super().__init__(model, quant_config, is_quantized)
        self.quant_config:QuIPConfig = quant_config
        self.search_result = None
                
    @classmethod
    def check_quant_config(self, quant_config: QuIPConfig):
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
        post_init: bool = True,
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
        quant_config = QuIPConfig.from_pretrained(quant_path, **cached_file_kwargs)

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

    def _prepare_examples_for_quantization(self, examples: List[Union[List[int], torch.LongTensor]], batch_size: int = 1, pad_token_id: Optional[int] = None):
        if pad_token_id is None:
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
            if pad_token_id is not None:
                examples = pad_sequence(new_examples, True, padding_value=pad_token_id)
            else:
                examples = torch.cat(new_examples,dim=0)
                
        if examples.dim() > 2:
            raise Exception('examples must be 2D tensor or less.')
        else:
            for i in range(examples.dim(),2):
                examples.squeeze(0)

        new_examples = []
        for start in range(0, examples.shape[0], batch_size):
            input_ids = examples[start: start + batch_size]
            new_examples.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
        return new_examples

    @torch.inference_mode()
    def quantize(
        self,
        examples: List[Union[List[int], torch.LongTensor]],
        pad_token_id: Optional[int] = None,
        batch_size: int = 1,
        cache_examples_on_gpu: bool = True,
    ):
        device_map = self.hf_device_map
        if device_map:
            for name, device in device_map.items():
                if device == "cpu":
                    logging.info(f"truly offloading {name} to cpu with hook.")
                    module = get_module_by_name_suffix(self.model, name)
                    remove_hook_from_module(module, recurse=True)
                    cpu_offload_with_hook(module, 'cuda:0')

        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []
        layer_outputs = []

        examples = self._prepare_examples_for_quantization(examples, batch_size, pad_token_id)

        class LayerHijacker(nn.Module):
            """hijack layer's forward pass to cache data"""

            def __init__(self, m, device):
                super().__init__()
                self.module = m
                self.data_device = device if cache_examples_on_gpu else 'cpu'

            def forward(self, inp=None, **kwargs):
                if inp is None:  # some models use all key-value arguments in forward pass call
                    for kwarg_name in ["hidden_states"]:
                        if kwarg_name in kwargs:
                            inp = kwargs[kwarg_name]
                            break
                layer_inputs.append(move_to_device(inp, self.data_device))
                attention_masks.append(kwargs["attention_mask"].to(self.data_device))
                pos_ids = kwargs.get("position_ids", None)
                if pos_ids is not None:
                    position_ids.append(move_to_device(pos_ids, self.data_device))
                one_kwargs = dict()
                for k, v in kwargs.items():  # make sure other arguments also be captured
                    if k not in ["hidden_states", "attention_mask", "position_ids"]:
                        if isinstance(v, torch.Tensor):
                            one_kwargs[k] = move_to_device(v, self.data_device)
                        else:
                            one_kwargs[k] = v
                layer_input_kwargs.append(one_kwargs)
                raise ValueError

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        num_batches = len(examples)
        layers = get_module_by_name_prefix(self.model, self.layers_block_name)

        force_layer_back_to_cpu = False
        if get_device(layers[0]) == 'cpu':
            layers[0] = layers[0].to('cuda:0')
            force_layer_back_to_cpu = True

        cur_layer_device = get_device(layers[0])
        ori_outside_layer_module_devices = {}
        for module_name in self.outside_layer_modules:
            module = get_module_by_name_prefix(self.model, module_name)

            if module is None:
                continue

            ori_outside_layer_module_devices[module_name] = get_device(module)
            if module is not None:
                move_to_device(module, cur_layer_device)

        # get inputs for first layer
        layers[0] = LayerHijacker(layers[0], cur_layer_device)
        for example in examples:
            for k, v in example.items():
                if len(v.shape) == 1:
                    v = v.unsqueeze(0)
                example[k] = move_to_device(v, cur_layer_device)
            try:
                self.model(**example)
            except ValueError:
                pass
        layers[0] = layers[0].module

        move_to_device(layers[0], 'cpu' if force_layer_back_to_cpu else cur_layer_device)
        for module_name in self.outside_layer_modules:
            module = get_module_by_name_prefix(self.model, module_name)
            if module is not None:
                move_to_device(module, ori_outside_layer_module_devices[module_name])

        torch.cuda.empty_cache()

        # resize attention mask and position ids for some special models
        attention_masks = self._resize_attention_mask(attention_masks)
        position_ids = self._resize_position_ids(position_ids)

        inside_layer_modules = self.inside_layer_modules
        if not self.quant_config.true_sequential:
            inside_layer_modules = [sum(inside_layer_modules, [])]
        quantizers = {}
        for i in tqdm(range(len(layers)), desc="QuIP Quantization"):
            layer = layers[i]
            force_layer_back_to_cpu = False
            if get_device(layer) == 'cpu':
                move_to_device(layer, 'cuda:0')
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            full = get_named_module(layer)
            for names in inside_layer_modules:
                subset = {n: full[n] for n in names}
                quip = {}
                for name in subset:
                    quip[name] = QuIP(subset[name])
                    quip[name].configure(self.quant_config.quant,
                                         self.quant_config.bits,
                                         self.quant_config.npasses,
                                         unbiased=self.quant_config.unbiased)
                    quip[name].quantizer.configure(self.quant_config.bits,
                                                   perchannel=True,
                                                   sym=False,
                                                   qfn=self.quant_config.qfn,
                                                   mse=False)
                def add_batch(name):
                    def tmp(_, inp, out):
                        quip[name].add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(num_batches):
                    layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                    layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
                    additional_layer_inputs = {"attention_mask": layer_attention_mask}
                    layer_position_ids = None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
                    if layer_position_ids is not None:
                        additional_layer_inputs["position_ids"] = layer_position_ids
                    for k, v in layer_input_kwargs[j].items():
                        if isinstance(v, torch.Tensor):
                            additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                        else:
                            additional_layer_inputs[k] = v
                    layer(layer_input, **additional_layer_inputs)
                    
                for h in handles:
                    h.remove()

                for name in subset:
                    quip[name].post_batch()

                for name in subset:
                    quip[name].preproc(preproc_gptqH=self.quant_config.pre_gptqH, percdamp=self.quant_config.damp_percent,
                                       preproc_rescale=self.quant_config.pre_rescale, preproc_proj=self.quant_config.pre_proj,
                                       preproc_proj_extra=self.quant_config.pre_proj_exta)
                    scale, zero, g_idx = quip[name].fasterquant(lazy_batch=self.quant_config.lazy_batch)
                    quantizers[f'{self.layers_block_name}.{i}.{name}'] = (
                        move_to_device(scale, 'cpu' if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(zero, 'cpu' if force_layer_back_to_cpu else cur_layer_device),
                        g_idx
                    )
                    quip[name].free()

            for j in range(num_batches):
                layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
                additional_layer_inputs = {"attention_mask": layer_attention_mask}
                layer_position_ids = None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
                if layer_position_ids is not None:
                    additional_layer_inputs["position_ids"] = layer_position_ids
                for k, v in layer_input_kwargs[j].items():
                    if isinstance(v, torch.Tensor):
                        additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                    else:
                        additional_layer_inputs[k] = v
                layer_output = move_to_device(
                    layer(layer_input, **additional_layer_inputs)[0],
                    cur_layer_device if cache_examples_on_gpu else 'cpu'
                )
                layer_outputs.append(layer_output)

            layers[i] = move_to_device(layer, 'cpu' if force_layer_back_to_cpu else cur_layer_device)
            del layer
            del quip
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            torch.cuda.empty_cache()

        self.pack_model(quantizers=quantizers)
        if device_map:
            self.model = remove_hook_from_module(self.model, recurse=True)
            self.model = simple_dispatch_model(self.model, device_map)
            
        self.model.config.use_cache = forward_pass_use_cache
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