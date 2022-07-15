_base_ = './queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_classification.py'
# optimizer
# optimizer = dict(_delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001)
optimizer = dict(_delete_=True, type='AdamW', lr=0.0000025, weight_decay=0.0001)
# learning policy
lr_config = dict(warmup_iters=1)