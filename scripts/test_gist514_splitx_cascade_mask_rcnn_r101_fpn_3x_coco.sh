python tools/test.py configs/ultra_sound/cascade_mask_rcnn/resnet101/gist514_split0_cascade_mask_rcnn_r101_fpn_3x_coco.py work_dirs/gist514_split0_cascade_mask_rcnn_r101_fpn_3x_coco/latest.pth --eval bbox vanilla --show-dir playground/usd/cascade_mask_rcnn/show_results/split0 2>&1 | tee work_dirs/gist514_split0_cascade_mask_rcnn_r101_fpn_3x_coco/test.log

python tools/test.py configs/ultra_sound/cascade_mask_rcnn/resnet101/gist514_split1_cascade_mask_rcnn_r101_fpn_3x_coco.py work_dirs/gist514_split1_cascade_mask_rcnn_r101_fpn_3x_coco/latest.pth --eval bbox vanilla --show-dir playground/usd/cascade_mask_rcnn/show_results/split1 2>&1 | tee work_dirs/gist514_split1_cascade_mask_rcnn_r101_fpn_3x_coco/test.log

python tools/test.py configs/ultra_sound/cascade_mask_rcnn/resnet101/gist514_split2_cascade_mask_rcnn_r101_fpn_3x_coco.py work_dirs/gist514_split2_cascade_mask_rcnn_r101_fpn_3x_coco/latest.pth --eval bbox vanilla --show-dir playground/usd/cascade_mask_rcnn/show_results/split2 2>&1 | tee work_dirs/gist514_split2_cascade_mask_rcnn_r101_fpn_3x_coco/test.log

python tools/test.py configs/ultra_sound/cascade_mask_rcnn/resnet101/gist514_split3_cascade_mask_rcnn_r101_fpn_3x_coco.py work_dirs/gist514_split3_cascade_mask_rcnn_r101_fpn_3x_coco/latest.pth --eval bbox vanilla --show-dir playground/usd/cascade_mask_rcnn/show_results/split3 2>&1 | tee work_dirs/gist514_split3_cascade_mask_rcnn_r101_fpn_3x_coco/test.log

python tools/test.py configs/ultra_sound/cascade_mask_rcnn/resnet101/gist514_split4_cascade_mask_rcnn_r101_fpn_3x_coco.py work_dirs/gist514_split4_cascade_mask_rcnn_r101_fpn_3x_coco/latest.pth --eval bbox vanilla --show-dir playground/usd/cascade_mask_rcnn/show_results/split4 2>&1 | tee work_dirs/gist514_split4_cascade_mask_rcnn_r101_fpn_3x_coco/test.log
