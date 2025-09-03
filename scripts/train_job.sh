#!/bin/bash
#SBATCH --job-name=exp_cub200
#SBATCH --partition=a40
#SBATCH --qos=a40
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
# bash scripts/coop/main.sh cub200 vit_b16_ep50 front 4 4 True   # dataset model position NCTX shots CSC

# CoCoOp
# bash scripts/cocoop/base2new_train.sh cub200 3 4                # dataset seed shots
# bash scripts/cocoop/base2new_test.sh cub200 1

# MaPLe
# bash scripts/maple/base2new_train_maple.sh cub200 3 4           # dataset seed shots
# bash scripts/maple/base2new_test_maple.sh cub200 3

# PromptSRC
# train and test on given dataset for K=1 shot over 3 seeds
bash scripts/promptsrc/few_shot.sh cub200 4                      # dataset shots

# PLOT
# bash scripts/plot/main.sh caltech101 4                                 # dataset no_of_prompts
# --> Rn50
# cd /home/vis-comp/24m2119/PLOT/plot-coop/scripts
# bash main.sh cub200 4 
# --> ViT
# cd /home/vis-comp/24m2119/PLOT/plot-pp/scripts
# bash main_visiononly.sh cub200 4
# bash main_joint.sh cub200 4
# bash evaluation.sh cub200 4

# DAPT
# cd /home/vis-comp/24m2119/DAPT/
# bash scripts/gen_prototype.sh 0
# bash scripts/main.sh cub200 0            # dataset gpu_id

# DePT
# cd /home/vis-comp/24m2119/DePT
# python parallel_runner.py --cfg maple_dept               # DePT/configs.py file is most useful --> change # shots

# KgCoOp
# cd /home/vis-comp/24m2119/KgCoOp
# bash scripts/base2new_train.sh cub200 8.0

# ProGrad
# cd /home/vis-comp/24m2119/KgCoOp
# bash scripts/base2new_train.sh caltech101 8.0
# bash scripts/base2new_test.sh caltech101 8.0

# TCP
# bash scripts/tcp/base2new_train.sh cub200

# LASP
# bash scripts/lasp/base2new_train.sh cub200 
# bash scripts/lasp/base2new_test.sh caltech101

# RPO
# bash scripts/rpo/base2new_train.sh cub200
# bash scripts/rpo/base2new_test.sh caltech101




