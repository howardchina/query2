_base_ = ['./gist514_split0_cascade_mask_rcnn_r101_fpn_3x_coco.py', ]

data_root = '/mnt/home1/workspace2/QueryInst/data/usd514_jpeg_roi'
split = 'split_2'
data = dict(
    train=dict(ann_file=data_root+'/annotations/train_anno_crop_'+split+'.json'),
    val=dict(ann_file=data_root+'/annotations/val_anno_crop_'+split+'.json'),
    test=dict(ann_file=data_root+'/annotations/val_anno_crop_'+split+'.json'))