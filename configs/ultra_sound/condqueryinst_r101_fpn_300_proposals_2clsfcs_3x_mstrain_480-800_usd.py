_base_ = './condqueryinst_r101_fpn_100_proposals_3x_mstrain_480-800_usd.py'

num_proposals = 300
num_stages = 6
load_from = 'work_dirs/pretrained/queryInst/queryinst_r101_300_queries-860dc5d5.pth'

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='CondDIIHead',
                num_classes=2,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=2, # 1 -> 2
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
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)]))