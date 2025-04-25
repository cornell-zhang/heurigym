#!/bin/bash
set -x
net="c432"
K=6
# iscas85 directory
abc -c "read dataset/iscas85/input/${net}.blif; if -K ${K}; write_blif dataset/iscas85/output/${net}_lut${K}_abc.blif"
abc -c "cec dataset/iscas85/input/${net}.blif dataset/iscas85/output/${net}_lut${K}_abc.blif"

# demo directory
# abc -c "read dataset/demo/input/${net}.blif; if -K ${K}; write_blif dataset/demo/output/${net}_lut${K}_abc.blif"
# abc -c "cec dataset/demo/input/c17.blif dataset/demo/output/c17_lut3_abc.blif"