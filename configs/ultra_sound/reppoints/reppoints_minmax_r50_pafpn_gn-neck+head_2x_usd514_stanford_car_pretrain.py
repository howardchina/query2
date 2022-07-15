_base_ = ['../../reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck+head_2x_coco.py',]

dataset_type = 'AnatomyDataset'
data_root = '/mnt/home1/workspace2/QueryInst/data/usd514_jpeg_roi'
split = 'split_1'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                   (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                   (1333, 736), (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnatomy'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='AnatomyDataset',
        ann_file=data_root+'/annotations/train_anno_crop_'+split+'.json',
        img_prefix=data_root+'/images/',
        pipeline=train_pipeline,
        classes=('lmym', 'GIST')),
    val=dict(
        type='AnatomyDataset',
        ann_file=data_root+'/annotations/val_anno_crop_'+split+'.json',
        img_prefix=data_root+'/images/',
        pipeline=test_pipeline,
        classes=('lmym', 'GIST')),
    test=dict(
        type='AnatomyDataset',
        ann_file=data_root+'/annotations/val_anno_crop_'+split+'.json',
        img_prefix=data_root+'/images/',
        pipeline=test_pipeline,
        classes=('lmym', 'GIST')))
evaluation = dict(metric=['bbox'])

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) # don't set lr=0.001, that is too small
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500, # often warmup is removed in finetune, but I kept it for faire comparison
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/mnt/home1/workspace2/QueryInst/work_dirs/reppoints_minmax_r50_pafpn_gn-neck+head_2x_stanford_car_no_coco_pretrain/epoch_24.pth'
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='RepPointsDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    bbox_head=dict(
        type='RepPointsHead',
        num_classes=2,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method='minmax',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg=dict(
        init=dict(
            assigner=dict(type='PointAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
gpu_ids = range(0, 4)