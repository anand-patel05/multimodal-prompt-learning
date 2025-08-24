feature_dir='clip_feat'

for DATASET in CUB200
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 3
done
