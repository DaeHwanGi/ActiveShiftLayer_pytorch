#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import MyDataset
from unet import UNet

import cv2
import numpy as np
# In[2]:


dataset_path = 'deepscores/deep_scores_v2_100p/images_png/'
gt_path = 'deepscores/deep_scores_v2_100p/pix_annotations_png/'

f = open("test_files.txt", 'r')
lines = f.readlines()
test_file_names =[]
for line in lines:
    test_file_names.append(line)
f.close()


# In[3]:


batch_size = 2
epochs = 20
size_x = 512 #540 #1080
size_y = 768 #960 #1920


# In[5]:


train_dataset = MyDataset(dataset_path, gt_path, test_file_names, size_x, size_y)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_classes=159, padding=True, up_mode='upconv').to(device)
model.load_state_dict(torch.load("pytorch_weight.pt"))
optim = torch.optim.Adam(model.parameters())

global_idx = 0
for ep in range(epochs):
    loss_list = []
    for X, y in dataloader:
        X = X.to(device)  # [N, 1, H, W]
        prediction = model(X)  # [N, 2, H, W]

        for idx in range(batch_size):
            prediction_np = torch.argmax(prediction, dim=1).cpu().numpy()
            cv2.imwrite("deepscores/test_output/{}.png".format(global_idx), prediction_np[idx])
            global_idx += 1




