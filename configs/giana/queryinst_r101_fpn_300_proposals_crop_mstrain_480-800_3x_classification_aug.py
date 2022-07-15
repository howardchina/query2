_base_ = '../queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'

load_from = 'work_dirs/pretrained/queryInst/queryinst_r101_300_queries-860dc5d5.pth'
# load_from = 'work_dirs/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_usd489/epoch_29.pth'

# load_from = 'work_dirs/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_usd/epoch_4.pth'
num_stages = 6
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=2,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    with_proj=True,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ],
        mask_head=[
            dict(
                type='DynamicMaskHead',
                num_classes=2,
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=14,
                    with_proj=False,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                dropout=0.0,
                num_convs=4,
                roi_feat_size=14,
                in_channels=256,
                conv_kernel_size=3,
                conv_out_channels=256,
                class_agnostic=False,
                norm_cfg=dict(type='BN'),
                upsample_cfg=dict(type='deconv', scale_factor=2),
                loss_dice=dict(type='DiceLoss', loss_weight=8.0)) for _ in range(num_stages)
        ]
    )
)



albu_train_transforms = [
    dict(
        type='Cutout',
        max_h_size=64,
        p=0.5),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.5),
    dict(
        type='RGBShift',
        r_shift_limit=10,
        g_shift_limit=10,
        b_shift_limit=10,
        p=0.5),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=10,
        sat_shift_limit=10,
        val_shift_limit=10,
        p=0.5),
    dict(type='JpegCompression', quality_lower=80, quality_upper=99, p=0.5),
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, value) for value in min_values],
        multiscale_mode='value',
        keep_ratio=True),
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], 
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]

dataset_type = 'CocoDataset'
classes = ('no adenoma', 'adenoma')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/giana/classification/annotations/train.json',
        img_prefix='data/giana/classification/m_train/m_train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/giana/classification/annotations/valid.json',
        img_prefix='data/giana/classification/m_valid/m_valid/images/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/giana/classification/annotations/valid.json',
        img_prefix='data/giana/classification/m_valid/m_valid/images/'))