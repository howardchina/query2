_base_ = './reppoints_minmax_r50_pafpn_gn-neck+head_1x_stanford_car_no_coco_pretrain.py'
load_from = 'work_dirs/reppoints_minmax_r50_pafpn_gn-neck+head_3x_stanford_car_no_coco_pretrain/epoch_10.pth'
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='step', step=[23], warmup=None)
total_epochs = 26
runner = dict(max_epochs=total_epochs)
checkpoint_config = dict(interval=5)
model = dict(# for faster validation. change it during formal test.
    test_cfg=dict(
        nms_pre=15, # keep top15 for faster nms
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=1))# keep top5