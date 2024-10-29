import time
import torch
import numpy as np

import mmcv

from itertools import zip_longest
from typing import Optional

# from mmengine.registry import MODELS
from mmengine.registry import MODELS, build_from_cfg
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)

import torch.nn as nn

from mmengine.model.base_model import BaseModel
from mmpose.models.data_preprocessors import PoseDataPreprocessor
from mmpose.models.builder import LOSSES, BACKBONES
from mmpose.models.backbones.base_backbone import BaseBackbone
# from .pct_tokenizer import Tokenizer ## PCT tokenizer for STAGE I
# from .pct_tokenizer import PCT_Tokenizer

from mmengine.structures import InstanceData, PixelData

from peft import LoraConfig, get_peft_model, TaskType


@BACKBONES.register_module()
class PeftBackbone(nn.Module):
    def __init__(self,
                 lora_cfg=None,
                 backbone=None,
                 ):
        super().__init__()
        self.lora_cfg = dict(
            peft_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            target_modules = ["q_proj", "v_proj"],
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
        )
        if lora_cfg is not None:
            self.lora_cfg.update(lora_cfg)
        peft_config = LoraConfig(**lora_cfg)
        
        base_model = build_from_cfg(backbone, MODELS)
        self.base_model = get_peft_model(base_model,peft_config=peft_config)
        
        
        def state_dict_hook(self, state_dict, prefix, local_metadata):
            assert "backbone.base_model" in prefix
            new_state_dict = {k.replace("backbone.base_model", "backbone"):v for k, v in state_dict.items() if "backbone.base_model" in k}
            return new_state_dict
        
        def load_state_dict_pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            state_dict_items = tuple(state_dict.items())
            for k, v in state_dict_items:
                for tgt in self.lora_cfg['target_modules']:
                    if f".{tgt}." not in k: continue
                    if f".{tgt}.lora_A" in k: continue
                        
                    # if state_dict has base_model layer without lora
                    new_key = k.replace(f".{tgt}.",f".{tgt}.base_layer.")
                    state_dict.update({new_key:v})
                    unexpected_keys.remove(k)
                    missing_keys.remove(new_key)
                    del state_dict[k]
                    
        self.base_model._register_state_dict_hook(state_dict_hook)
        self.base_model._register_load_state_dict_pre_hook(load_state_dict_pre_hook)

    
    def __getattr_(self, name: str):
        return getattr(self.base_model, name)
    
    def __call__(self,*args, **kargs):
        return self.base_model.__call__(*args, **kargs)
        
    