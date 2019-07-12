import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import cv2
from ActiveShift2d import ActiveShift2d
import numpy as np

batch_size = 128
learning_rate = 0.002
num_epoch = 100

cifar_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
cifar_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            ActiveShift2d(8),
            nn.Conv2d(8, 16, 1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 14 x 14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 x 7 x 7

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(7)
        )
        self.fc_layer = nn.Linear(256, 10)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)

        return out

model = CNN()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()

colors = np.random.choice(np.arange(0.5, 1, 0.01), size=(16,3))
for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image
        y_ = label

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

        print(loss)
        padd = 20
        show_img = image[0].repeat(3,1,1).permute(1,2,0).numpy()
        show_img = cv2.resize(show_img, dsize=(280, 280), interpolation=cv2.INTER_LINEAR)
        show_img = cv2.copyMakeBorder(show_img, padd, padd, padd, padd, cv2.BORDER_CONSTANT, value=(0,0,0))
        for k in range(8):
            alpha = int((model.layer[2].theta[k,0].item()+1)/2 * 27 * 10) + padd
            beta = int((model.layer[2].theta[k,1].item()+1)/2 * 27 * 10) + padd
            show_img = cv2.rectangle(show_img, (alpha, beta), (alpha+280, beta+280), list(colors[k]), 1)
        cv2.imshow("test", show_img)
        k = cv2.waitKey(1)
        if k == 27:  # esc key
            cv2.destroyAllWindow()