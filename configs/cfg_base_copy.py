_base_ = ["./default_runtime.py", "./ap10k.py"]
work_dir = "swin_lora_wrapper"

scope = "mmpose"
_scope_ = "mmpose"
default_scope = "mmpose"

# runtime
max_epochs = 90
base_lr = 1e-4

train_cfg = dict(max_epochs=max_epochs, val_interval=5)
randomness = dict(seed=21)


log_level = "INFO"
load_from = None

checkpoint_config = dict(interval=5, create_symlink=False)
evaluation = dict(interval=5, metric="mAP", save_best="AP")

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type="Adam",
        lr=base_lr,
    )
)
# learning rate
param_scheduler = [
    # warm-up
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    # Use a cosine learning rate at [100, 900) iterations
    dict(type="CosineAnnealingLR", T_max=1500, by_epoch=False, begin=500, end=1600),
]
# codec settings
codec = dict(type="MSRAHeatmap", input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="PeftBackbone",
        # pretrained="pretrained/swin_b_p4_w7_coco_256x192-7432be9e_20220705.pth",
        lora_cfg=dict(
            r=8,
            lora_alpha=32,
            lora_dropout=0.03,
            bias="none",
            target_modules=["qkv", "proj"],
        ),
        backbone=dict(
            type="SwinTransformer",
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(3,),
            with_cp=False,
            convert_weights=True,
            frozen_stages=-1,
        ),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=1024,
        out_channels=17,
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode="heatmap",
        shift_heatmap=True,
    ),
    init_cfg=dict(
        type="Pretrained",
        checkpoint="pretrained/swin_b_p4_w7_coco_256x192-7432be9e_20220705.pth",
    ),
)

# base dataset settings
dataset_type = "AP10KDataset"
data_mode = "topdown"

backend_args = dict(backend="local")
find_unused_parameters = True

# pipelines
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]


val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

test_pipeline = val_pipeline


# data loaders
data_root = "data/aptv2/"
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    # persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/train_annotations.json",
        data_prefix=dict(img="data/"),
        pipeline=train_pipeline,
        metainfo=dict(from_file="configs/ap10k.py"),
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    # persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/val_annotations.json",
        data_prefix=dict(img="data/"),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file="configs/ap10k.py"),
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/val_annotations.json",
        data_prefix=dict(img="data/"),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file="configs/ap10k.py"),
    ),
)


# evaluators
val_evaluator = dict(
    type="CocoMetric",
    use_area=True,
    ann_file=f"{data_root}/annotations/val_annotations.json",
)
test_evaluator = dict(
    type="CocoMetric",
    use_area=True,
    ann_file=f"{data_root}/annotations/val_annotations.json",
)

val_cfg = dict()
test_cfg = dict()
