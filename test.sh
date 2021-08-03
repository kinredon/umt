#!/bin/bash
set -e

name=${2}
echo $name
net=$(echo $name | sed 's/models\/\([^\/:]*\).*/\1/g')
dataset=$(echo $name | sed 's/.*target\_\(.*\)\_session.*/\1/g')
for i in $(seq 1 8)
do    
    model_name=${name%epoch*}epoch_${i}_step${name#*step}
    CUDA_VISIBLE_DEVICES=${1} python umt_test.py --dataset $dataset --net $net --load_name $model_name 2>&1 | tee log/$(basename $model_name).log
done
