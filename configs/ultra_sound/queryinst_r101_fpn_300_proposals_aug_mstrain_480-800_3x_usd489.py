_base_ = './queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_usd489.py'

load_from = 'work_dirs/pretrained/queryInst/queryinst_r101_300_queries-860dc5d5.pth'

albu_train_transforms = [
    # dict(
    #     type='Cutout',
    #     max_h_size=64,
    #     p=0.5),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=1,
        p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.5),
    # dict(
    #     type='RGBShift',
    #     r_shift_limit=10,
    #     g_shift_limit=10,
    #     b_shift_limit=10,
    #     p=0.5),
    # dict(
    #     type='HueSaturationValue',
    #     hue_shift_limit=10,
    #     sat_shift_limit=10,
    #     val_shift_limit=10,
    #     p=0.5),
    # dict(type='JpegCompression', quality_lower=80, quality_upper=99, p=0.5),
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

data = dict(train=dict(pipeline=train_pipeline))