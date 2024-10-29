
import torch
import torch.nn as nn

from typing import Optional
from peft import LoraConfig, get_peft_model, TaskType

from mmpose.registry import MODELS

from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)

@MODELS.register_module()
class LoraTopdownPoseEstimator(TopdownPoseEstimator):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None,
                 lora_cfg: Optional[dict] = None,
                 ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)
        
        peft_config = LoraConfig(peft_type=TaskType.FEATURE_EXTRACTION, 
                                 target_modules= lora_cfg['target_modules'],
                                 r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], lora_dropout=lora_cfg['dropout'],bias=lora_cfg['bias'])
        
        self.backbone = get_peft_model(self.backbone, peft_config=peft_config)
        
        if lora_cfg['pretrained'] is not None:
            pth = torch.load(lora_cfg['pretrained'])
            new_state_dict = {k[len('backbone.'):]:v for k, v in pth['state_dict'].items() if k in "lora_"}
            self.backbone.load_state_dict(new_state_dict, strict=False)
        
        def hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            items = list(state_dict.items())
            for k, v in items:
                if "lora_" in k:
                    print("lora state_dict not be loaded here.")
                    continue
                
                new_key = prefix + "base_model.model." + k[len(prefix):]
                
                # add 'base_layer' of lora target layer
                for tgt in lora_cfg['target_modules']:
                    if f".{tgt}." in k:
                        new_key = new_key.replace(tgt, f"{tgt}.base_layer")
                state_dict.update({new_key:v})
                del state_dict[k]
                           
        
        self.backbone._register_load_state_dict_pre_hook(hook)
        self.register_state_dict_pre_hook()
        
        