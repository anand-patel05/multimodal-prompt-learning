# sh feat_extractor.sh
DATA="/home/vis-comp/24m2119/multimodal-prompt-learning/datasets/all_datasets"
OUTPUT='/home/vis-comp/24m2119/multimodal-prompt-learning/lpclip/clip_feat/'
SEED=1
SUB=all

# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet cub200
for DATASET in cub200
do
    for SPLIT in train val test
    do
        python feat_extractor.py \
        --split ${SPLIT} \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file ../configs/datasets/${DATASET}.yaml \
        --config-file ../configs/trainers/CoOp/vit_b16_val.yaml \
        --output-dir ${OUTPUT} \
        --eval-only
    done
done
