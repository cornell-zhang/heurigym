#!/bin/bash
set -x
python program/main.py --net dataset/iscas85/blif/c17.blif --k 3 --optimize size --out dataset/iscas85/output/c17_k3_size.blif 
