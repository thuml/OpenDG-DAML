#!/usr/bin/env bash
# Validate

cd ..

gpu=0

source=ACP
target=R

python validate.py \
    --root /data3/trans_lib/officehome \
    -d OfficeHome \
    -s ${source} \
    -t ${target} \
    --gpu ${gpu}