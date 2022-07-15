_base_=['../../fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py', ]


dataset_type = 'AnatomyDataset'
data_root = '/mnt/home1/workspace2/QueryInst/data/usd514_jpeg_roi'
split = 'split_0'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='Affine', translate_percent=0., rotate=90, fit_output=True, p=0.25),
            dict(type='Affine', translate_percent=0., rotate=180, fit_output=True, p=0.25),
            dict(type='Affine', translate_percent=0., rotate=270, fit_output=True, p=0.25)
        ],
        p=0.75),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadAnatomy'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                   (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                   (1333, 736), (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='FormatAnatomyBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'])
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
            dict(type='FormatAnatomyBundle'),
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

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=10)


load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth'


# model settings
model = dict(bbox_head=dict(num_classes=2))