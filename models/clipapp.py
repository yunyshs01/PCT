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



class ClipBackbone(nn.Module):
    def __init__(self,
                 model_name = None,
                 ):
        super().__init__()
        
        def save_intermediate_state(module : nn.Module, _, output):
            module.register_buffer("output_transformer", output.permute(1,0,2))
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model, _ = clip.load(self.model_name, device = self.device)
        
        self.hook = self.model.visual.transformer.register_forward_hook(save_intermediate_state)
        
        
    
    def forward(self, img):
        # img [B, 3, H, W]
        # print(img.shape)
        # exit()
        with torch.no_grad():
            x = self.model.encode_image(img)
        
        
        out = self.model.visual.transformer.output_transformer
        # out : [B, 1 + 49, D]
        return out
    
    def encode_text(self, text_token):
        # text_token : List[Tensor]  | Tensor
        
        with torch.no_grad():
            out = self.model.encode_text(text_token)
        
        return out
        
        
        
        


@MODELS.register_module()
class ClipAlign(BaseModel):
    def __init__(self,
                 data_preprocessor = None,
                 init_cfg = None,
                 cfg = None,
                 ):
        super().__init__(data_preprocessor, init_cfg)
        
        self.test_cfg = cfg['test_cfg']
        self.pct_pretrained = cfg['tokenizer']['ckpt']
        
        
        self.tokenizer = build_from_cfg(dict(type = "Tokenizer",stage_pct = "classifier", tokenizer=cfg['tokenizer']),MODELS)
        self._load_tokenizer(self.pct_pretrained)
        
        self.token_dim = cfg['tokenizer']['codebook']['token_dim']
        self.token_num = cfg['tokenizer']['codebook']['token_num']
        self.codebook_num = cfg['tokenizer']['codebook']['token_class_num']
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip = ClipBackbone(model_name = cfg['model_name'])
        self.vf_dim = self.clip.model.visual.positional_embedding.data.shape[-1] # 768
        
        
        self.dim_feat = self.clip.model.positional_embedding.data.shape[-1] # 512
        
        self.num_keypoints = cfg['num_keypoints']
        self.kpt_loss = build_from_cfg(cfg['kpt_loss'], MODELS)
        # self.cls_loss = build_from_cfg(cfg['cls_loss'],MODELS)
        self.cls_loss = nn.CrossEntropyLoss(reduction = 'mean')
    
        
        
        
        
        self.fc_category = nn.Linear(self.vf_dim, self.dim_feat)
        self.proj = nn.Linear(self.vf_dim, self.dim_feat, bias=False)
        
        self.TR_IMG = 49
        
        self.blocks = nn.ModuleList([
            MixerLayer(self.dim_feat,self.dim_feat,self.TR_IMG,self.TR_IMG,0.1) for _ in range(3)
        ])
        
        self.ln = nn.LayerNorm(self.dim_feat)
        
        self.fc_token = nn.Conv1d(self.TR_IMG,self.token_num,kernel_size=1)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_feat,self.dim_feat),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.dim_feat, self.codebook_num)
        )
           
        
        
    def forward(self, inputs, data_samples, mode):
        img, kpt_token, cat_token = inputs
        
        B = img.shape[0]
        V = self.vf_dim
        D = self.dim_feat
        K = self.num_keypoints
        
        # img : preprocessed image [B,3, H, W]
        #kpt_token : keypoint tokens [K, L]
        #cat_token : category tokens [B, L]
        
        
        gt_kpt = self.clip.encode_text(kpt_token).float().view(1,K,1,D)
        # gt_cls = self.clip.encode_text(cat_token).float().view(B,1,1,D)
        #gt_kpt [1, K, 1, Dt] : keypoint feature from dataset keypoints name
        #gt_cls [B, 1, 1, Dt] : text feature from dataset category name
        
        
        vis_feature = self.clip(img).float().view(B, 1, -1, V) 
        #vis_feature = [B, 1, 1 + 49, D]
        
        # print(gt_kpt.shape)
        # exit()
        
        #[B, 1, 1, Dv], [B, 1, 49, Dv]
        cls_feature, img_feature = torch.split(vis_feature, [1, self.TR_IMG], dim = 2) 

        # cls_feature = self.fc_category(cls_feature)
        #cls_feature [B, 1, 1, Dt]
        
        losses = {}
        # if mode == "loss":
        #     mse = self.cls_loss(cls_feature, gt_cls)
        #     losses.update(cls_mse_loss = mse)
        
        img_feature = self.proj(img_feature).view(B, 1, -1, D)  # [B, 1, 49, D]
        
        
        # # TODO
        # sim = F.cosine_similarity(img_feature, gt_kpt , dim=-1) # [B, K, 49]
        
        # _, indicies = topk = sim.topk(1,dim = 2)
        # # indicies [B,K,1]
        
        # batch_indices = torch.arange(B).unsqueeze(1).expand(B,K)
        # indicies = indicies.squeeze(-1)
        
        # top_feature = img_feature.squeeze(1)[batch_indices,indicies]
        
        # top_feature : B, K, D
        
        top_feature = img_feature.squeeze(1)
        
        for block in self.blocks:
            top_feature = block(top_feature)
            
        top_feature = self.fc_token(top_feature)
        # [B, T, D]
        
        top_feature = self.mlp(top_feature)
        # [B, T, Ncd]
    
        
        out = self._decode_feature(top_feature)
        
        
        
        
        if mode == "loss":
            
        
            gt = [d.gt_instances.get("transformed_keypoints", None) for d in data_samples]
            gt = np.stack(gt)
            gt = torch.from_numpy(gt).float().view(B, K, 2).to(img.device)
                
            visible  = [d.gt_instances.get("keypoints_visible", None) for d in data_samples]
            visible = np.stack(visible)
            visible = torch.from_numpy(visible > 0.5).float().view(B, K, 1).to(img.device)
            
            gt = torch.cat([gt,visible],dim = -1)
            
            
            
            with torch.no_grad():
                encodings, gt_indicies, _ = self.tokenizer(gt)
            
            cls_loss = self.cls_loss(top_feature,gt_indicies.detach())
            kpt_loss = self.kpt_loss(out, gt)
            

            indicies = indicies.view(B, self.token_num)
            gt_indicies = gt_indicies.view(B, self.token_num)
            cls_acc = indicies.eq(gt_indicies).float().mean(1).mean(0).detach().cpu()
            
            losses.update(cls_acc = cls_acc, cls_loss = cls_loss, kpt_loss = kpt_loss)
            
            
            return losses
               
        
        if mode == "predict" or True:
            recovered_joints = out.detach().cpu().numpy()
        
            if isinstance(recovered_joints, tuple):
                batch_pred_instances, batch_pred_fields = recovered_joints
            else:
                batch_pred_instances = recovered_joints
                batch_pred_fields = None

            results = self.add_pred_to_datasample(batch_pred_instances, batch_pred_fields, data_samples)

            return results
        
    def _load_tokenizer(self,pretrained_path):
        pt = torch.load(pretrained_path)
        state_dict = pt['state_dict'].copy()
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith("tokenizer."):
                new_key = key.replace("tokenizer.","")
                state_dict.update({new_key:state_dict[key]})
                del state_dict[key]
            
        self.tokenizer.load_state_dict(state_dict=state_dict,strict=True)
    
    def _decode_feature(self, top_features : torch.Tensor):
        # top_feature : [B, T, Ncd]
        
        top_features = F.softmax(top_features,dim = -1)
        top_features = top_features.view(-1, self.codebook_num)
        
        with torch.no_grad():
            
            part_token_feat = torch.matmul(top_features, self.tokenizer.codebook)
            # [B, T, Dtkn]
            
            decoder = self.tokenizer.decoder
            
            part_token_feat = part_token_feat.view(-1, self.token_num, self.token_dim)
            part_token_feat = part_token_feat.transpose(2,1)
            part_token_feat = decoder.token_mlp(part_token_feat).transpose(2,1)
            decode_feat = decoder.decoder_start(part_token_feat)
            
            
            for num_layer in decoder.decoder_layers:
                decode_feat = num_layer(decode_feat)
            decode_feat = decoder.layer_norm(decode_feat)

            recoverd_joints = decoder.recover_embed(decode_feat)
        # [B, K, 2]
            
        return recoverd_joints
            
        
        
    
    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """

        # 임시 스코어
        sc = np.ones((17, ))


        # print(len(batch_pred_instances), len(batch_data_samples))
        assert len(batch_pred_instances) == len(batch_data_samples)
        
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        #for pred_instances, pred_fields, data_sample in zip_longest(
        for prediction, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']
            input_size = data_sample.metainfo['input_size']

            ##
            pred_instances = InstanceData()
            # exapand instance dimension (17, 2) => (1, 17, 2)
            pred_instances.set_field(np.expand_dims(prediction, axis=0), "keypoints")
            pred_instances.set_field(np.expand_dims(sc, axis=0), "keypoint_scores")

            pred_instances.keypoints[..., :2] = \
                pred_instances.keypoints[..., :2] / input_size * input_scale \
                + input_center - 0.5 * input_scale
            
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)


            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices], key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
    
            
        
        
@MODELS.register_module()
class ClipPreprocess(ImgDataPreprocessor):
    def __init__(self,
                 dataset_info = None,
                 mean = None,
                 std = None,
                 non_blocking = False,
                 *args,
                 **kargs
                 ):
        super().__init__(mean = mean, std = std, non_blocking=non_blocking, *args, **kargs)
        self.dataset_info = dataset_info
        
        
    def forward(self, data: dict, training: bool = False) -> dict | list:
        data = super().forward(data)
        
        # keypoint text features
        kpt_names = self._get_kpt_names()
        K = len(kpt_names)        
        template = "The {} in the photo"
        prompts = [template.format(kpt) for kpt in kpt_names] # [K,]
        kpt_token = clip.tokenize(prompts).to(data['inputs'].device)
        
        
        
        # category text features
        if training:
            category = [d.metainfo.get('category', "object") for d in data['data_samples']] #[B, ]
            cat_token = clip.tokenize(category).to(data['inputs'].device)
        else:
            cat_token = None
        
        img = data['inputs']
        
        data['inputs'] =  img, kpt_token, cat_token
        
        return data
            
        
        
    
    def _get_kpt_names(self):
        
        kpt_info = self.dataset_info['keypoint_info']
        names = []
        for i in range(len(kpt_info)):
            name = kpt_info[i]['name']
            name.replace("L_","Left ")
            name.replace("R_","Right ")
            name.replace("F_","Front ")
            name.replace("B_","Back ")
            names.append(name)
        return names
            
        
        