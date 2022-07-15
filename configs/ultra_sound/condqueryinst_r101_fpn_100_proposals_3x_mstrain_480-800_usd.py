_base_ = './condqueryinst_r50_fpn_100_proposals_1x_mstrain_480-800_usd.py'

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

lr_config = dict(policy='step', step=[27, 33])
total_epochs = 36
runner = dict(max_epochs=total_epochs)