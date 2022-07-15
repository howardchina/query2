_base_ = '../queryinst/queryinst_r50_fpn_1x_coco.py'

load_from = 'work_dirs/queryinst_r50_fpn_1x_coco/epoch_12.pth'
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

data_root = 'data/usd/'
dataset_type = 'CocoDataset'
classes = ('GIST', 'LMYM')
data = dict(
    # samples_per_gpu=2,
    # workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'images/'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/'))


# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0000025, weight_decay=0.0001)
# optimizer
optimizer = dict(type='SGD', lr=0.000025, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))