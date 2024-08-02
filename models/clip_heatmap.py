import torch
import torch.nn as nn

import torch.nn.functional as F
from itertools import zip_longest

import numpy as np

from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from typing import Optional

from mmengine.model.base_model import BaseModel, ImgDataPreprocessor
from mmengine.registry import MODELS, build_from_cfg
from mmengine.structures import InstanceData



import clip

from clip.model import Transformer

from .modules import MixerLayer
# from .tokenizer_modified import Tokenizer

from einops import rearrange

@MODELS.register_module()
class ClipHeatmap(BaseModel):
    def __init__(self,
                 data_preprocessor = None,
                 backbone = None,
                 init_cfg = None,
                 cfg = None,
                 
                 ):
        super().__init__(data_preprocessor,init_cfg)
        
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        # self.backbone = self._load_clip_backbone(cfg['clip_model']).to(DEVICE)
        self.backbone = build_from_cfg(backbone,MODELS)
        
        
        self.deconv_channels = cfg.get("deconv_channels", [(768,384),(384,192),(192,96),(96,48)])
        self.kernel_size = 4
        self.stride = 2
        self.img_patch_shape = cfg['img_patch_shape']
        self.num_keypoints = cfg.get("num_keypoints",17)
        
        self.deconv_blocks = nn.ModuleList([
            # nn.Sequential(
            #     nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
            #                        kernel_size=self.kernel_size, stride = self.stride,padding = 1),
            #     nn.BatchNorm2d(out_channels),
            #     nn.ReLU(),
            # ) for in_channels, out_channels in self.deconv_channels
        ])
        
        self.keypoint_head = build_from_cfg(cfg['head'],MODELS)
        
        
        
    
    def _load_clip_backbone(self,clip_model):
        model, _ = clip.load(clip_model)
        
        
        
        
        vit =  model.visual
        
        def hook(module : nn.Module, _, output):
            module.register_buffer("final",output.permute(1,0,2))
        vit.transformer.register_forward_hook(hook)
        
        return model
        
    
    def forward(self, inputs, data_samples, mode):
        # inputs : B, 3, 224, 224
        B = inputs.shape[0]
        K = self.num_keypoints
        
        
        # with torch.no_grad():
        #     self.backbone.visual(inputs.half())
        # out_feature = self.backbone.visual.transformer.final[:,1:].float()
        
        out_feature = self.backbone(inputs)[-1].contiguous()
        # print(out_feature.shape)
        # exit()
        # out_feature = rearrange(out_feature,"b (w h) c -> b c w h", h = self.img_patch_shape[0]).contiguous()
        
        
        # assert out_feature.shape[-1] == self.deconv_channels[0][0]
        
        for block in self.deconv_blocks:
            out_feature = block(out_feature)
            
        
        if mode == "loss":
            losses = {}
            
            loss = self.keypoint_head.loss([out_feature],data_samples)
            losses.update(loss)
            
            return losses
        
        if mode == "predict":
            
            preds = self.keypoint_head.predict([out_feature], data_samples)
            for data_sample, pred in zip(data_samples, preds):
                data_sample.pred_instances = pred
            return data_samples

        
            
        
        
        
            
            
        
        
        
        
        
        
        
        
        
        
        