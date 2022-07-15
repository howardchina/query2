_base_ = './reppoints_minmax_r50_pafpn_gn-neck+head_1x_stanford_car_no_coco_pretrain.py'
lr_config = dict(policy='step', step=[10, 33])
total_epochs = 36
runner = dict(max_epochs=total_epochs)
checkpoint_config = dict(interval=5)
# model = dict(
#     test_cfg=dict(
#         nms_pre=15, # keep top15 for faster nms
#         min_bbox_size=0,
#         score_thr=0.05,
#         nms=dict(type='nms', iou_threshold=0.5),
#         max_per_img=5))# keep top5