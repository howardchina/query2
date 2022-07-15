 CUDA_VISIBLE_DEVICES=0 python ./tools/train.py configs/ultra_sound/detr/50/gist514_split0_detr_r50_8x2_150e_coco.py --seed 666  &
 CUDA_VISIBLE_DEVICES=1 python ./tools/train.py configs/ultra_sound/detr/50/gist514_split1_detr_r50_8x2_150e_coco.py --seed 666  &
 CUDA_VISIBLE_DEVICES=2 python ./tools/train.py configs/ultra_sound/detr/50/gist514_split2_detr_r50_8x2_150e_coco.py --seed 666  &
 CUDA_VISIBLE_DEVICES=3 python ./tools/train.py configs/ultra_sound/detr/50/gist514_split3_detr_r50_8x2_150e_coco.py --seed 666 
 CUDA_VISIBLE_DEVICES=3 python ./tools/train.py configs/ultra_sound/detr/50/gist514_split4_detr_r50_8x2_150e_coco.py --seed 666 