_base_ = './reppoints_minmax_r50_pafpn_gn-neck+head_1x_stanford_car_no_coco_pretrain.py'
lr_config = dict(step=[11, 26])
runner = dict(type='EpochBasedRunner', max_epochs=30)
model = dict(# for faster validation. change it during formal test.
    test_cfg=dict(
        nms_pre=15, # keep top15 for faster nms
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=1))# keep top5
model = dict(
    bbox_head=dict(
        loss_bbox_init=dict(_delete_=True, type='GIoULoss', loss_weight=0.5),
        loss_bbox_refine=dict(_delete_=True, type='GIoULoss', loss_weight=1.0)))