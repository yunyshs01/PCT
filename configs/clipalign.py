_base_ = ['./default_runtime.py', './ap10k.py']


scope = 'mmpose'

# runtime
max_epochs = 100
base_lr = 1e-2

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.15 ),

    # paramwise_cfg=dict(
    #     norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
    )

# learning rate
param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=1.0e-5,
    #     by_epoch=False,
    #     begin=0,
    #     end=30),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        # begin=max_epochs // 2,
        # end=max_epochs,
        # T_max=max_epochs // 2,
        by_epoch=True,
        # convert_to_iter_based=True
        ),
]

# codec settings

#Keypoints codec
codec = dict(
    type='SimCCLabel', 
    input_size=(224, 224),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=True,
    use_dark=False)

#Heatmap

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])


# model settings
model = dict(
    type='ClipAlign',
    # data preprocessor
    data_preprocessor=dict(
        type = "ClipPreprocess",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        dataset_info = {{_base_.dataset_info}},
        bgr_to_rgb=True),
    cfg = dict(
        num_keypoints = 17,
        kpt_loss = dict(type = "JointS1Loss",beta = 1.),
        model_name = "ViT-B/32",
        tokenizer=dict(
            # guide_ratio=0.0,
            ckpt="work_dirs/my_base_tokenizer/epoch_50.pth",
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
            ),
        test_cfg = dict(
            flip_test=False,
            dataset_name='AP10K')
        ),
        
        
        
        
    )
    



# base dataset settings
dataset_type = 'AP10KDataset'
data_mode = 'topdown'
data_root = 'data/ap10k/'

backend_args = dict(backend='local')
find_unused_parameters=True

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),

    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(
        type='TopdownAffine', 
        input_size=codec['input_size'], 
        use_udp=True
        ),
    # dict(type='mmdet.YOLOXHSVRandomAug'),
    # dict(
    #     type='Albumentation',
    #     transforms=[
    #         dict(type='Blur', p=0.1),
    #         dict(type='MedianBlur', p=0.1),
    #         dict(
    #             type='CoarseDropout',
    #             max_holes=1,
    #             max_height=0.4,
    #             max_width=0.4,
    #             min_holes=1,
    #             min_height=0.2,
    #             min_width=0.2,
    #             p=1.0),
    #     ]),


    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        pack_transformed=True
    )
]


val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='TopdownAffine', 
        input_size=codec['input_size'], 
        use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        pack_transformed=True
    )
]

test_pipeline = val_pipeline



# data loaders
data_root = 'data/apt36k'
train_dataloader = dict(
    batch_size=7,
    num_workers=3,
    pin_memory = True,
    # persistent_workers=True,
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
    # persistent_workers=True,
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
    batch_size=512,
    num_workers=8,
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
