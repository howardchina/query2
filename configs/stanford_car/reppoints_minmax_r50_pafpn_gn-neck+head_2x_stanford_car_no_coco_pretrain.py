_base_ = './reppoints_minmax_r50_pafpn_gn-neck+head_1x_stanford_car_no_coco_pretrain.py'
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
