_base_ = ['./default_runtime.py', './ap10k.py']


scope = 'mmpose'
_scope_ = 'mmpose'
default_scope = 'mmpose'

# runtime
max_epochs = 50
base_lr = 5e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=5)
randomness = dict(seed=21)


log_level = 'INFO'
load_from = None

find_unused_parameters=False
evaluation = dict(interval=5, metric='mAP', save_best='AP')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    # paramwise_cfg=dict(
    #     norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
    )

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=40, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]
# codec settings
codec = dict(
    type='SimCCLabel', 
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=True,
    use_dark=False)

# model settings
model = dict(
    type='PCT',
    # data preprocessor
    data_preprocessor=dict(
        _scope_='mmpose',
        type='PoseDataPreprocessor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        bgr_to_rgb=True),
    # pretrained='weights/simmim/swin_base.pth',
    # backbone
    backbone=dict(
        type='SwinV2TransformerRPE2FC',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[16, 16, 16, 8],
        pretrain_window_size=[12, 12, 12, 6],
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
        rpe_interpolation='geo',
        use_shift=[True, True, False, False],
        relative_coords_table_type='norm8_log_bylayer',
        attn_type='cosine_mh',
        rpe_output_type='sigmoid',
        postnorm=True,
        mlp_type='normal',
        out_indices=(3,),
        patch_embed_type='normal',
        patch_merge_type='normal',
        strid16=False,
        frozen_stages=5,
    ),
    head=dict(
        type='Tokenizer',
        stage_pct='tokenizer',
        in_channels=1024,
        image_size=codec['input_size'],
        num_joints=17,
        # loss_keypoint=dict(
        #     type='Classifer_loss',
        #     token_loss=1.0,
        #     joint_loss=1.0),
        # cls_head=dict(
        #     conv_num_blocks=2,
        #     conv_channels=256,
        #     dilation=1,
        #     num_blocks=4,
        #     hidden_dim=64,
        #     token_inter_dim=64,
        #     hidden_inter_dim=256,
        #     dropout=0.0),
        tokenizer=dict(
            # guide_ratio=0.0,
            # ckpt="",
            encoder=dict(
                drop_rate=0.2,
                num_blocks=4,
                hidden_dim=512,
                token_inter_dim=64,
                hidden_inter_dim=512,
                dropout=0.0,
            ),
            decoder=dict(
                num_blocks=1,
                hidden_dim=32,
                token_inter_dim=64,
                hidden_inter_dim=64,
                dropout=0.0,
            ),
            codebook=dict(
                token_num=34,
                token_dim=512,
                token_class_num=4096,
                ema_decay=0.9,
            ),
            loss_keypoint=dict(
                type='Tokenizer_loss',
                joint_loss_w=1.0, 
                e_loss_w=15.0,
                beta=0.05,)
            )),
    test_cfg=dict(
        flip_test=False,
        dataset_name='AP10K'))


# base dataset settings
dataset_type = 'AP10KDataset'
data_mode = 'topdown'
data_root = 'data/ap10k/'

backend_args = dict(backend='local')


# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs',
         pack_transformed=True,
         )
]


val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs',
         pack_transformed=True,
         )
]

test_pipeline = val_pipeline




# data loaders
data_root = 'data/apt36k'
train_dataloader = dict(
    batch_size=1024,
    num_workers=8,
    pin_memory = True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, 
        data_root=data_root,
        ann_file='annotations/train_annotations_1.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
        metainfo=dict(from_file='configs/ap10k.py')
    ))

val_dataloader = dict(
    batch_size=1024,
    num_workers=8,
    pin_memory = True,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val_annotations_1.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/ap10k.py')
    ))

test_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val_annotations_1.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/ap10k.py')
    ))



# evaluators
val_evaluator = dict(
    type='CocoMetric',
    use_area=True,
    ann_file=f'{data_root}/annotations/val_annotations_1.json')
test_evaluator = dict(
    type='CocoMetric',
    use_area=True,
    ann_file=f'{data_root}/annotations/val_annotations_1.json')

val_cfg = dict()
test_cfg = dict()
