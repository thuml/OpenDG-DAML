#!/usr/bin/env bash
# OfficeHome

cd ..

gpu=0

ipe=101

source=ACP
target=R

log_name=daml-${source}-${target}

python daml.py \
    --root /data3/trans_lib/officehome \
    -d OfficeHome \
    -s ${source} \
    -t ${target} \
    --gpu ${gpu} \
    --iters_per_epoch ${ipe} \
    --savename ${log_name}