_base_ = './reppoints_minmax_r50_pafpn_gn-neck+head_2x_stanford_car_no_coco_pretrain.py'
# @heqi use GIoU loss to take place of smooth l1 loss
model = dict(
    bbox_head=dict(
        loss_bbox_init=dict(type='GIoULoss', loss_weight=0.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=1.0)))