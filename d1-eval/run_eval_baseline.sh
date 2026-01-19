#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "========== Starting LLaDA-1.5 Evaluation =========="
python -m torch.distributed.run \
  --nproc_per_node 8 \
  --master_port 29411 \
  eval.py \
  --dataset countdown \
  --batch_size 8 \
  --gen_length 128 \
  --block_length 64 \
  --output_dir eval_results_baseline \
  --model_path "GSAI-ML/LLaDA-1.5"

echo "========== Starting LLaDA-8B-Instruct Evaluation =========="
python -m torch.distributed.run \
  --nproc_per_node 8 \
  --master_port 29412 \
  eval.py \
  --dataset countdown \
  --batch_size 12 \
  --gen_length 128 \
  --block_length 64 \
  --output_dir eval_results_baseline \
  --model_path "GSAI-ML/LLaDA-8B-Instruct"

echo "========== Parsing Results =========="
# revise the last sentence of parse_and_get_acc.py to evaluate folder eval_results_baseline
python ./parse_and_get_acc.py