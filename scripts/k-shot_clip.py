# This is k-shot prototype based CLIP Implementation

import json
import random
import torch
import clip
from PIL import Image
from collections import defaultdict
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# Load JSON file
json_path = "/home/vis-comp/24m2119/multimodal-prompt-learning/datasets/all_datasets/cub200/split_zhou_CUB200.json"
with open(json_path, "r") as f:
    data = json.load(f)

train_data = data["train"]
test_data = data["test"]

# ------------------------------
# Get k samples from each class
# ------------------------------
def get_k_shot_samples(train_data, k, seed):
    random.seed(seed)   # fix seed here for reproducibility
    class_to_samples = defaultdict(list)

    for img_path, class_id, class_name in train_data:
        class_to_samples[class_id].append((img_path, class_name))

    k_shot_data = {}
    for class_id, samples in class_to_samples.items():
        if len(samples) < k:
            raise ValueError(f"Class {class_id} has only {len(samples)} samples, less than k={k}")
        k_shot_data[class_id] = random.sample(samples, k)

    return k_shot_data

# ---------------------------
# Compute prototype for each class
# ---------------------------
def compute_prototypes(k_shot_data, img_root="/home/vis-comp/24m2119/fine-grained_classfication/images"):
    prototypes = {}
    for class_id, samples in k_shot_data.items():
        embeddings = []
        for path, class_name in samples:
            img = preprocess(Image.open(f"{img_root}/{path}")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())
        prototypes[class_id] = np.mean(embeddings, axis=0)
    return prototypes

# ---------------------------
# Classify test samples
# ---------------------------
def evaluate(test_data, prototypes, img_root="/home/vis-comp/24m2119/fine-grained_classfication/images"):
    correct = 0
    total = 0
    for path, class_id, class_name in test_data:
        img = preprocess(Image.open(f"{img_root}/{path}")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        sims = {cid: (emb.cpu().numpy() @ proto.T).item() for cid, proto in prototypes.items()}
        pred = max(sims, key=sims.get)

        if pred == class_id:
            correct += 1
        total += 1

    return correct / total * 100

# ---------------------------
# Main loop: average over seeds
# ---------------------------
if __name__=="__main__":
    seeds = [1, 2, 3]
    k_shots = [1, 2, 4, 8, 16]

    for k in k_shots:
        acc_list = []
        for seed in seeds:
            k_shot_data = get_k_shot_samples(train_data, k, seed)
            prototypes = compute_prototypes(k_shot_data)
            acc = evaluate(test_data, prototypes)
            acc_list.append(acc)

        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        print(f"{k}-shot accuracy: {mean_acc:.2f}% ± {std_acc:.2f}")