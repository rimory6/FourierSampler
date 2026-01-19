#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "========== Starting LLaDA-1.5 Evaluation =========="
python -m torch.distributed.run \
  --nproc_per_node 8 \
  --master_port 29411 \
  eval.py \
  --dataset countdown \
  --batch_size 3 \
  --gen_length 128 \
  --block_length 64 \
  --output_dir eval_results_fft \
  --model_path "GSAI-ML/LLaDA-1.5" \
  --use_fft \
  --fft_window_ratio 0.2 \
  --fft_beta_min 0.4 \
  --fft_beta_max 0.6

echo "========== Starting LLaDA-8B-Instruct Evaluation =========="
python -m torch.distributed.run \
  --nproc_per_node 8 \
  --master_port 29412 \
  eval.py \
  --dataset countdown \
  --batch_size 9 \
  --gen_length 128 \
  --block_length 64 \
  --output_dir eval_results_fft \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --use_fft \
  --fft_window_ratio 0.4 \
  --fft_beta_min 0.4 \
  --fft_beta_max 0.6

echo "========== Parsing Results =========="
# revise parse_and_get_acc.py to evaluate folder eval_results_fft
python ./parse_and_get_acc.py