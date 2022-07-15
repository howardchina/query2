_base_ = './reppoints_minmax_r50_pafpn_gn-neck+head_2x_stanford_car_no_coco_pretrain.py'
optimizer = dict(_delete_=True, type='Adam', lr=0.01, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=1, norm_type=2))
lr_config = dict(step=[16, 22])
