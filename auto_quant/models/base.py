import os
import json
import logging
from typing import Optional, Union
from dataclasses import field, fields

import torch
import torch.nn as nn
from safetensors.torch import save_file as safe_save

from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils.hub import PushToHubMixin, cached_file, create_repo, create_commit, CommitOperationAdd
from transformers.modeling_utils import shard_checkpoint, WEIGHTS_NAME, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from accelerate import init_empty_weights, infer_auto_device_map

from auto_quant import __version__
from auto_quant.modules.qlinear_exllama import check_exllama_can_save

class BaseQuantConfig(PushToHubMixin):
    quant_path: Optional[str] = None
    quant_type = None
    
    def save_pretrained(self, save_dir: str, **kwargs):
        with open(os.path.join(save_dir, "quant_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

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
        
        field_names = [field.name for field in fields(self)]
        with open(resolved_config_file, "r", encoding="utf-8") as f:
            args_from_json = json.load(f)
            if args_from_json['quant_type'] != self.quant_type:
                raise ValueError(f"Expected quant type: {self.quant_type}, quant type in config: {args_from_json['quant_type']}")
            
            filtered_args = {}
            for key, val in args_from_json.items():
                if key in field_names:
                    filtered_args[key] = val
                else:
                    if key not in ['quant_type']:
                        logging.warning(f"ignoring unknown parameter in {quant_config_filename}: {key}.")
            return self(**filtered_args)
            
class BaseQuantForCausalLM(nn.Module, PushToHubMixin, GenerationMixin):
    def __init__(self, model, quant_config, is_quantized):
        super().__init__()
        self.model:PreTrainedModel = model
        self.config = self.model.config
        self.model_type:str = self.model.config.model_type
        self.is_quantized:bool = is_quantized
        if not isinstance(quant_config, BaseQuantConfig):
            raise Exception('Please use QuantConfig, which Inheritance BaseQuantConfig.')
        self.quant_config = quant_config

    @property
    def device(self):
        if not self.hf_device_map:
            return self.model.device
        else:
            device = [d for d in self.hf_device_map.values() if d not in {'cpu', 'disk'}][0]
            return torch.device(device)

    @property
    def hf_device_map(self):
        return getattr(self.model, "hf_device_map", None)
    
    def to(self, device: Union[str, torch.device]):
        self.model.to(device)
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self, *args, **kwargs):
        return self.model.get_input_embeddings(*args, **kwargs)
            
    def save_pretrained(self, save_dir: str, max_shard_size: Union[int, str] = "10GB", use_safetensors: bool = False):
        """alias of save_quantized"""
        logging.warning("you are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir, max_shard_size)

    def save_quantized(self, save_dir: str, max_shard_size: Union[int, str] = "10GB", use_safetensors: bool = False):    
        if not self.is_quantized:
            raise EnvironmentError("can only save quantized model, please execute .quantize first.")
        if not check_exllama_can_save(self.model):
            raise EnvironmentError("desc_act is enabled and Exllama is post-initialized. The model cannot be saved.")
        
        os.makedirs(save_dir, exist_ok=True)        
        
        state_dict = self.model.state_dict()
        state_dict = {k: v.to('cpu') for k, v in state_dict.items()}
        shards, index = shard_checkpoint(state_dict, max_shard_size, weights_name=SAFE_WEIGHTS_NAME if use_safetensors else WEIGHTS_NAME)
        
        for shard_file, shard in shards.items():
            if use_safetensors:
                safe_save(shard, os.path.join(save_dir, shard_file), {'format': "pt", 'auto_quant_version': str(__version__)})
            else:
                torch.save(shard, os.path.join(save_dir, shard_file))
        if index is not None:
            save_index_file = os.path.join(save_dir, SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            
        self.model.config.save_pretrained(save_dir)
        self.quant_config.save_pretrained(save_dir)
        self.quant_config.quant_path = save_dir
        
    @classmethod
    def from_pretrained(
        self, 
        model_path, 
        quant_config, 
        max_memory: Optional[dict] = None,
        torch_dtype: torch.dtype = torch.float16, 
        trust_remote_code: bool = False,
        **model_init_kwargs,
    ):  
        from .auto import AutoQuantConfig
        if isinstance(quant_config, AutoQuantConfig):
            quant_config = quant_config.quant_config
        
        self.check_quant_config(quant_config)
        # Parameters related to loading from Hugging Face Hub
        cache_dir = model_init_kwargs.pop("cache_dir", None)
        force_download = model_init_kwargs.pop("force_download", False)
        resume_download = model_init_kwargs.pop("resume_download", False)
        proxies = model_init_kwargs.pop("proxies", None)
        local_files_only = model_init_kwargs.pop("local_files_only", False)
        use_auth_token = model_init_kwargs.pop("use_auth_token", None)
        revision = model_init_kwargs.pop("revision", None)
        subfolder = model_init_kwargs.pop("subfolder", "")
        commit_hash = model_init_kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
        }
    
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, **cached_file_kwargs)
        model_init_kwargs["torch_dtype"] = torch_dtype
        model_init_kwargs["trust_remote_code"] = trust_remote_code
        if max_memory:
            if "disk" in max_memory:
                raise NotImplementedError("disk offload not support yet.")
                
            with accelerate.init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype, trust_remote_code=True)
            model.tie_weights()
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=[self.layer_type],
                low_zero=False
            )
            model_init_kwargs["device_map"] = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[self.layer_type],
            )
            model_init_kwargs["low_cpu_mem_usage"] = True
            del model
        else:
            model_init_kwargs["device_map"] = None
            model_init_kwargs["low_cpu_mem_usage"] = False
        torch.cuda.empty_cache()
        merged_kwargs = {**model_init_kwargs, **cached_file_kwargs}
        model = AutoModelForCausalLM.from_pretrained(model_path, **merged_kwargs)
        model.eval()
        return self(model, quant_config, is_quantized=False)
        
    def push_to_hub(
        self,
        repo_id: str,
        save_dir: Optional[str] = None,
        max_shard_size: Union[int, str] = "10GB",
        use_safetensors: bool = False,
        commit_message: Optional[str] = "Upload of AutoQuant quantized model",
        use_auth_token: Optional[Union[bool, str]] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: Optional[bool] = False,
    ) -> str:
        """
        Upload the model to the Hugging Face Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your tool to. It should contain your organization name when
                pushing to a given organization.
            save_dir (`str`, *optional*):
                The name of the local folder to save the model to.
                If the model has already been saved, this parameter can be omitted.
            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. 
                Checkpoints shard will then be each of size lower than this size.
            commit_message (`str`, *optional*, defaults to `"Upload of AutoQuant quantized model"`):
                Message to commit while pushing.
            use_auth_token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
        """
        if (self.quant_config.quant_path is None or not os.path.isdir(self.quant_config.quant_path)) and save_dir is None:
            raise ValueError("Quantized model should be saved first, or you can provide save_dir to make sure model is saved to local disk before uploading.")
        
        if save_dir is not None:
            logging.info(f"Saving model to {save_dir}")
            self.save_quantized(save_dir, max_shard_size)
            
        repo_url = create_repo(
            repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="model"
        )
        repo_id = repo_url.repo_id

        if self.quant_config.quant_path is not None:
            work_dir = self.quant_config.quant_path
            operations = [
                CommitOperationAdd(path_or_fileobj=os.path.join(work_dir, f), path_in_repo=f)
                for f in os.listdir(work_dir)
            ]
            logging.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=commit_message,
                token=use_auth_token,
                create_pr=create_pr,
                repo_type="model",
            )
        
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            return getattr(self.model, item)