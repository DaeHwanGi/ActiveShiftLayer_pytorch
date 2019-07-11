import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model import AS_ResNet, BasicBlock

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    ])
test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    ])

cifar_train = dset.CIFAR100("./", train=True, transform=train_transform, target_transform=None, download=True)
cifar_test = dset.CIFAR100("./", train=False, transform=test_transform, target_transform=None, download=True)

batch_size = 128
learning_rate = 0.002
num_epoch = 100

train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=batch_size, shuffle=False)

model = AS_ResNet(BasicBlock, [1,3,4,6,3], num_classes=100)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()
for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image
        y_ = label

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

        print("\r[{}/{}] loss : {:.4f}".format(j, len(train_loader), loss.item()), end="")
