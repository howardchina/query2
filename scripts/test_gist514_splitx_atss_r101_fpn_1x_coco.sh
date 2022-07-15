python tools/test.py configs/ultra_sound/atss/101/gist514_split0_atss_r101_fpn_1x_coco.py work_dirs/gist514_split0_atss_r101_fpn_1x_coco/latest.pth --eval bbox vanilla --show-dir playground/usd/gist514_split0_atss_r101_fpn_1x_coco/show_results/split0 2>&1 | tee work_dirs/gist514_split0_atss_r101_fpn_1x_coco/test.log

python tools/test.py configs/ultra_sound/atss/101/gist514_split1_atss_r101_fpn_1x_coco.py work_dirs/gist514_split1_atss_r101_fpn_1x_coco/latest.pth --eval bbox vanilla --show-dir playground/usd/gist514_split1_atss_r101_fpn_1x_coco/show_results/split1 2>&1 | tee work_dirs/gist514_split1_atss_r101_fpn_1x_coco/test.log

python tools/test.py configs/ultra_sound/atss/101/gist514_split2_atss_r101_fpn_1x_coco.py work_dirs/gist514_split2_atss_r101_fpn_1x_coco/latest.pth --eval bbox vanilla --show-dir playground/usd/gist514_split2_atss_r101_fpn_1x_coco/show_results/split2 2>&1 | tee work_dirs/gist514_split2_atss_r101_fpn_1x_coco/test.log

python tools/test.py configs/ultra_sound/atss/101/gist514_split3_atss_r101_fpn_1x_coco.py work_dirs/gist514_split3_atss_r101_fpn_1x_coco/latest.pth --eval bbox vanilla --show-dir playground/usd/gist514_split3_atss_r101_fpn_1x_coco/show_results/split3 2>&1 | tee work_dirs/gist514_split3_atss_r101_fpn_1x_coco/test.log

python tools/test.py configs/ultra_sound/atss/101/gist514_split4_atss_r101_fpn_1x_coco.py work_dirs/gist514_split4_atss_r101_fpn_1x_coco/latest.pth --eval bbox vanilla --show-dir playground/usd/gist514_split4_atss_r101_fpn_1x_coco/show_results/split4 2>&1 | tee work_dirs/gist514_split4_atss_r101_fpn_1x_coco/test.log
