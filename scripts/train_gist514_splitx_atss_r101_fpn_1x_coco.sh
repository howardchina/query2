 CUDA_VISIBLE_DEVICES=0 python ./tools/train.py configs/ultra_sound/atss/101/gist514_split0_atss_r101_fpn_1x_coco.py --seed 666  &
 CUDA_VISIBLE_DEVICES=1 python ./tools/train.py configs/ultra_sound/atss/101/gist514_split1_atss_r101_fpn_1x_coco.py --seed 666  
 CUDA_VISIBLE_DEVICES=0 python ./tools/train.py configs/ultra_sound/atss/101/gist514_split2_atss_r101_fpn_1x_coco.py --seed 666  &
 CUDA_VISIBLE_DEVICES=1 python ./tools/train.py configs/ultra_sound/atss/101/gist514_split3_atss_r101_fpn_1x_coco.py --seed 666  &
 CUDA_VISIBLE_DEVICES=2 python ./tools/train.py configs/ultra_sound/atss/101/gist514_split4_atss_r101_fpn_1x_coco.py --seed 666
 # CUDA_VISIBLE_DEVICES=0 python ./tools/train.py configs/ultra_sound/atss/101/gist514_split4_atss_r101_fpn_1x_coco.py --seed 666 