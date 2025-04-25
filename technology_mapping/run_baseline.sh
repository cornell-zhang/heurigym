#!/bin/bash
set -x
net="c432"
K=6
# iscas85 directory
python program/main.py --net dataset/iscas85/input/${net}.blif --k ${K} --optimize size --out dataset/iscas85/output/${net}_lut${K}_baseline.blif
abc -c "cec dataset/iscas85/input/${net}.blif dataset/iscas85/output/${net}_lut${K}_baseline.blif"


# demo directory
# python program/main.py --net dataset/demo/input/${net}.blif --k ${K} --optimize size --out dataset/demo/output/${net}_lut${K}_baseline.blif 
# abc -c "cec dataset/demo/input/${net}.blif dataset/demo/output/${net}_lut${K}_baseline.blif"