_base_ = './deformable_detr_r50_16x2_50e_stanford_cars_tiny.py'
lr_config = dict(policy='step', step=[120])
runner = dict(type='EpochBasedRunner', max_epochs=150)
