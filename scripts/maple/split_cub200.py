# import os

# root = "/home/vis-comp/24m2119/multimodal-prompt-learning/datasets/all_datasets/cub200/200_bird_species"  # path to your dataset folders

# NEW_CNAMES = {}
# for folder in os.listdir(root):
#     if os.path.isdir(os.path.join(root, folder)):
#         # Remove numeric prefix (001., 002., etc.)
#         clean_name = folder.split(".", 1)[1] if "." in folder else folder
#         # Lowercase + underscores
#         clean_name = clean_name.replace("_", " ").lower()
#         NEW_CNAMES[folder] = clean_name

# print(NEW_CNAMES)

# ==============================================================================================================================

# import json

# # Load JSON
# path = "/home/vis-comp/24m2119/multimodal-prompt-learning/datasets/all_datasets/caltech-101/split_zhou_Caltech101.json"
# with open(path, "r") as f:
#     data = json.load(f)

# # Get top-level keys
# print("Top-level keys:", data.keys())

# # Pretty-print first 2 levels
# import pprint
# pprint.pprint({k: type(v) for k, v in data.items()})

# ==============================================================================================================================

# import json

# def print_structure(obj, indent=0):
#     prefix = "  " * indent
#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             print(f"{prefix}{k}: {type(v).__name__}")
#             print_structure(v, indent + 1)
#     elif isinstance(obj, list):
#         print(f"{prefix}List[{len(obj)}]")
#         if len(obj) > 0:
#             print_structure(obj[0], indent + 1)

# with open(path, "r") as f:
#     data = json.load(f)

# print_structure(data)

# ==============================================================================================================================
# Note our split for CUB200 was created automatically bu CoOp code by zhou_split.
import os
import json
import random
from sklearn.model_selection import train_test_split

# Path to your dataset (change accordingly)
data_dir = "/path/to/CUB_200_2011/images"

# Get all class folders
classes = sorted(os.listdir(data_dir))

# Build mapping of class -> index
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

# Collect all (path, label, class_name)
all_samples = []
for cls_name in classes:
    cls_dir = os.path.join(data_dir, cls_name)
    images = os.listdir(cls_dir)
    for img in images:
        img_path = os.path.join(cls_name, img)  # relative path
        label = class_to_idx[cls_name]
        all_samples.append([img_path, label, cls_name])

print(f"Total samples: {len(all_samples)}")

# Split: 70% train, 15% val, 15% test
train_data, temp_data = train_test_split(all_samples, test_size=0.30, stratify=[x[1] for x in all_samples], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.50, stratify=[x[1] for x in temp_data], random_state=42)

# Final dictionary
dataset_split = {
    "train": train_data,
    "val": val_data,
    "test": test_data
}

# Save JSON
with open("cub200_split.json", "w") as f:
    json.dump(dataset_split, f, indent=4)

print("Saved split to cub200_split.json")
print("train:", len(train_data), "val:", len(val_data), "test:", len(test_data))



