import numpy as np
import torch
from mmpose.models.pose_estimators import TopdownPoseEstimator
import torch.nn as nn
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)

from mmengine.registry import MODELS, build_from_cfg
from torch import Tensor

class OccClassifier(nn.Module):
    def __init__(self,
                 cfg=None,
                 ):
        super().__init__()
        self.num_classes = cfg['num_classes']
        self.criterion = nn.BCEWithLogitsLoss()
        
    
        def get_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2, stride=2)
            )
    
        
        # Deconvolution layers
        self.conv_layers = nn.ModuleList([
            get_block(768, 256),
            get_block(256, 256),
            # get_block(256, 256),
            # get_block(256, 256),
        ])
        
        # self.pooling = nn.AdaptiveAvgPool2d((16,16))
        
        self.head = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, image_feat):
        feat = image_feat
        B = feat.shape[0]
        
        for conv_layer in self.conv_layers:
            feat = conv_layer(feat)
        
        # img : B x 768 x 1 x 1
        
        # Pooling layer
        # feat = self.pooling(feat)  # B x 16 x 16
        
        # Fully connected layer
        feat = feat.view(B, -1)  # B x 768
        pred = self.head(feat)  # B x num_classes
       
        return pred
    
    
    def loss(self, feat, data_samples):
        target = [d.gt_instances.get("keypoints_visible",None) for d in data_samples]
        target = np.stack(target)
        target = torch.from_numpy(target).float().to(feat.device)
        loss = self.criterion(pred, target)
        losses = dict(loss_occ=loss)
        return losses
        
            
            
            
@MODELS.register_module()   
class OccEstimator(TopdownPoseEstimator):
    def __init__(self,occ_cfg=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.occ = OccClassifier(cfg=occ_cfg)
    
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats = self.extract_feat(inputs)

        losses = dict()
        
        occ_losses= self.occ.loss(feats, data_samples)
        losses.update(occ_losses)

        if self.with_head:
            losses.update(
                self.head.loss(feats, data_samples, train_cfg=self.train_cfg))

        return losses


if __name__== "__main__":
    x = torch.randn(2, 768, 16, 16).to('cuda')
    model = OccClassifier(dict(num_classes=17, loss = dict(type="OccLoss"))).to('cuda')
    pred = model(x)
    y  = torch.randn(2, 17).to('cuda')
    
    # loss = model.loss(pred, [torch.randn(17) for _ in range(2)])
    
    # print(loss.detach().cpu().numpy().item())
    
        
        
        
        
        
