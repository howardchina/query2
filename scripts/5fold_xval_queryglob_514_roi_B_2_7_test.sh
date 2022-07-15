python tools/test.py configs/ultra_sound/queryglob/queryglob_usdanno514roi_B_2_7_split0.py work_dirs/queryglob_usdanno514roi_B_2_7_split0/epoch_36.pth --eval bbox segm glob 2>&1 | tee work_dirs/queryglob_usdanno514roi_B_2_7_split0/test_log_queryglob_usdanno514roi_B_2_7_split0.txt

python tools/test.py configs/ultra_sound/queryglob/queryglob_usdanno514roi_B_2_7_split1.py work_dirs/queryglob_usdanno514roi_B_2_7_split1/epoch_36.pth --eval bbox segm glob 2>&1 | tee work_dirs/queryglob_usdanno514roi_B_2_7_split1/test_log_queryglob_usdanno514roi_B_2_7_split1.txt

python tools/test.py configs/ultra_sound/queryglob/queryglob_usdanno514roi_B_2_7_split2.py work_dirs/queryglob_usdanno514roi_B_2_7_split2/epoch_36.pth --eval bbox segm glob 2>&1 | tee work_dirs/queryglob_usdanno514roi_B_2_7_split2/test_log_queryglob_usdanno514roi_B_2_7_split2.txt

python tools/test.py configs/ultra_sound/queryglob/queryglob_usdanno514roi_B_2_7_split3.py work_dirs/queryglob_usdanno514roi_B_2_7_split3/epoch_36.pth --eval bbox segm glob 2>&1 | tee work_dirs/queryglob_usdanno514roi_B_2_7_split3/test_log_queryglob_usdanno514roi_B_2_7_split3.txt

python tools/test.py configs/ultra_sound/queryglob/queryglob_usdanno514roi_B_2_7_split4.py work_dirs/queryglob_usdanno514roi_B_2_7_split4/epoch_36.pth --eval bbox segm glob 2>&1 | tee work_dirs/queryglob_usdanno514roi_B_2_7_split4/test_log_queryglob_usdanno514roi_B_2_7_split4.txt