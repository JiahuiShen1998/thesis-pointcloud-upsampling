#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="-1"
export PU_NET_FORCE_CPU=1
exec /home/ra87racy/miniconda3/envs/punet_tf/bin/python "$@"
