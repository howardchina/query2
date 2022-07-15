 CUDA_VISIBLE_DEVICES=0 python ./tools/train.py configs/ultra_sound/mask_rcnn/resnet101/gist514_split0_mask_rcnn_r101_fpn_3x_coco.py --seed 666  &
 CUDA_VISIBLE_DEVICES=1 python ./tools/train.py configs/ultra_sound/mask_rcnn/resnet101/gist514_split1_mask_rcnn_r101_fpn_3x_coco.py --seed 666  &
 CUDA_VISIBLE_DEVICES=2 python ./tools/train.py configs/ultra_sound/mask_rcnn/resnet101/gist514_split2_mask_rcnn_r101_fpn_3x_coco.py --seed 666  &
 CUDA_VISIBLE_DEVICES=3 python ./tools/train.py configs/ultra_sound/mask_rcnn/resnet101/gist514_split3_mask_rcnn_r101_fpn_3x_coco.py --seed 666 
 CUDA_VISIBLE_DEVICES=3 python ./tools/train.py configs/ultra_sound/mask_rcnn/resnet101/gist514_split4_mask_rcnn_r101_fpn_3x_coco.py --seed 666 