#  CUDA_VISIBLE_DEVICES=0 python ./tools/train.py configs/ultra_sound/query2/r101/gist514_split0_query2_r101_300pp.py --seed 666 

 ./tools/dist_train.sh configs/ultra_sound/query2/r101/gist514_split0_query2_r101_300pp.py 4 --no-validate --seed 666 
 ./tools/dist_train.sh configs/ultra_sound/query2/r101/gist514_split1_query2_r101_300pp.py 4 --no-validate --seed 666 
 ./tools/dist_train.sh configs/ultra_sound/query2/r101/gist514_split2_query2_r101_300pp.py 4 --no-validate --seed 666 
 ./tools/dist_train.sh configs/ultra_sound/query2/r101/gist514_split3_query2_r101_300pp.py 4 --no-validate --seed 666 
 ./tools/dist_train.sh configs/ultra_sound/query2/r101/gist514_split4_query2_r101_300pp.py 4 --no-validate --seed 666 