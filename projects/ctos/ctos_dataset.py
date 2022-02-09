import torch
import pandas as pd
from sklearn.model_selection import GroupKFold
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
import ast
from tqdm import tqdm

import detect
import train
import export
import val

# --------------------------- #
# Main configuration
# --------------------------- #
INPUT_IMAGE_SHAPE = (720, 1280, 3)
MODEL_INPUT_SHAPE = (1280, 1280, 3)
MAX_DETECTIONS = 50
BATCH_SIZE = 2
NUM_EPOCHS = 1
FINETUNE_EPOCHS = 1
KFOLD = 3
VAL_FOLD = 2
SEED = 2022


def parse_bbox(annotation):
    # Read annotation bboxes
    annotation_list = ast.literal_eval(annotation)
    (H, W) = INPUT_IMAGE_SHAPE[:2]
    bboxes_gt = []
    for annotation_item in annotation_list:
        x_min = max(0, annotation_item['x'])/W
        y_min = max(0, annotation_item['y'])/H
        x_max = min(W - 1, annotation_item['x'] + annotation_item['width'])/W
        y_max = min(H - 1, annotation_item['y'] + annotation_item['height'])/H
        bboxes_gt.append([(x_max+x_min)/2, (y_max+y_min)/2, x_max-x_min, y_max-y_min])
    return bboxes_gt

def get_image(image_path):
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# --------------------------- #
# Read dataset
# --------------------------- #
image_dir = "E:/Datasets/tensorflow-great-barrier-reef/train_images"
csv_path = "E:/Datasets/tensorflow-great-barrier-reef/train.csv"
df = pd.read_csv(csv_path)
df['image_path'] = image_dir + "/video_" + df.video_id.astype('string') + '/' + df.video_frame.astype('string') + ".jpg"
df = df[df['annotations'] != "[]"]
# Split train and validation by groupby
kf = GroupKFold(n_splits=KFOLD)
df = df.reset_index(drop=True)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(df, y=df.video_id.tolist(), groups=df.video_id)):
    df.loc[val_idx, 'fold'] = fold
train_df = df[df['fold'] != VAL_FOLD]
val_df = df[df['fold'] == VAL_FOLD]


# --------------------------- #
# Create dataset for training
# --------------------------- #
# train dataset
new_image_dir = "E:/Datasets/tensorflow-great-barrier-reef/train/images"
new_label_dir = "E:/Datasets/tensorflow-great-barrier-reef/train/labels"
if os.path.exists(new_image_dir):
    shutil.rmtree(new_image_dir)
if os.path.exists(new_label_dir):
    shutil.rmtree(new_label_dir)
os.makedirs(new_image_dir, exist_ok=True)
os.makedirs(new_label_dir, exist_ok=True)
images_path = train_df['image_path'].values
labels = train_df['annotations'].values
for i,image_path in tqdm(enumerate(images_path)):
    image_name_extension = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_name_extension)
    shutil.copyfile(image_path, new_image_dir+"/"+image_name_extension)
    bboxes = parse_bbox(labels[i])
    with open(new_label_dir+"/"+image_name+".txt", 'w') as outfile:
        for bbox in bboxes:
            outfile.write("0 "+str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3]) + "\n")


# val dataset
new_image_dir = "E:/Datasets/tensorflow-great-barrier-reef/val/images"
new_label_dir = "E:/Datasets/tensorflow-great-barrier-reef/val/labels"
if os.path.exists(new_image_dir):
    shutil.rmtree(new_image_dir)
if os.path.exists(new_label_dir):
    shutil.rmtree(new_label_dir)
os.makedirs(new_image_dir, exist_ok=True)
os.makedirs(new_label_dir, exist_ok=True)
images_path = val_df['image_path'].values
labels = val_df['annotations'].values
for i,image_path in tqdm(enumerate(images_path)):
    image_name_extension = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_name_extension)
    shutil.copyfile(image_path, new_image_dir+"/"+image_name_extension)
    bboxes = parse_bbox(labels[i])
    with open(new_label_dir+"/"+image_name+".txt", 'w') as outfile:
        for bbox in bboxes:
            outfile.write("0 "+str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3]) + "\n")

