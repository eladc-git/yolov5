import torch
import pandas as pd
from sklearn.model_selection import GroupKFold
import cv2
import numpy as np
import matplotlib.pyplot as plt

import detect
import train
import export
import val

# --------------------------- #
# Main configuration
# --------------------------- #
SINGLE = True
EVALUATION = False
TRACKER = False
EVALUATION_TRACKER = False
KFOLD = 3
VAL_FOLD = 2

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

images_path = val_df['image_path'].values
annotations = val_df['annotations'].values
sequences = val_df['sequence'].values

# Model
#model = torch.hub.load('.', 'custom', path=r"D:/model_out/weights/best.pt", source='local', force_reload=True)
model = torch.hub.load('.', 'custom', path=r"D:/models/best.pt", source='local')
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'D:/models/best_saved_model')

model.conf = 0.01
model.iou = 0.1
model.max_det = 20

# Images
index = 280
image_path = images_path[index]

# Inference
image = get_image(image_path)
results = model(image, size=3600, augment=True)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

bboxes = results.pred[0][:, :4].numpy().astype(np.int32)
scores = results.pred[0][:, 4].numpy()
image = get_image(image_path)

# Prediction
for i, box in enumerate(bboxes):
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=1)
    cv2.putText(image, text="ctos {:0.0f}%".format(100 * scores[i]), org=(box[0], box[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0), thickness=1)

cv2.imwrite("mygraph.jpg", image)
# plt.imshow(image)
# plt.show()