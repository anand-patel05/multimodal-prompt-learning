#!/bin/bash
#SBATCH --job-name=lpclip_cub200
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

conda activate /home/vis-comp/24m2119/miniconda3/envs/fine-grained
cd /home/vis-comp/24m2119/multimodal-prompt-learning

# zero-shot CLIP 
# bash scripts/zsclip/zeroshot.sh cub200 vit_b16                  # dataset model

# k-shot prototype based CLIP
# python ./scripts/k-shot_clip.py

# LP-CLIP
# cd ./lpclip
# bash feat_extractor.sh
# bash linear_probe.sh

# VPT
# bash scripts/vpt/base2new_train_vpt.sh cub200 3 4                 # dataset seed shots

# CoOp
# bash scripts/coop/main.sh cub200 vit_b16_ep50 front 16 4 True   # dataset model position CT shots CSC

# CoCoOp
# bash scripts/cocoop/base2new_train.sh cub200 3 4                # dataset seed shots
# bash scripts/cocoop/base2new_test.sh cub200 1

# MaPLe
# bash scripts/maple/base2new_train_maple.sh cub200 3 4           # dataset seed shots
# bash scripts/maple/base2new_test_maple.sh cub200 3

# PromptSRC
# train and test on given dataset for K=1 shot over 3 seeds
# bash scripts/promptsrc/few_shot.sh cub200 4                      # dataset shots
