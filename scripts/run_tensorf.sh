#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensorf.py data/nerf_synthetic/lego --workspace trial_tensorf
