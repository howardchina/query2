_base_ = './cascade_mask_rcnn_r101_fpn_20e_usd489.py'

load_from = './work_dirs/pretrained/cascade_rcnn/cascade_mask_rcnn_r101_fpn_20e_coco_bbox_mAP-0.434__segm_mAP-0.378_20200504_174836-005947da.pth'
# learning policy
lr_config = dict(policy='step', step=[27, 33], warmup_iters=1000)
total_epochs = 36
runner = dict(max_epochs=total_epochs)