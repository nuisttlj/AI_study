import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F


class CenterLoss(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn((cls_num, feature_num)).cuda())

    def forward(self, xs, ys):
        xs = F.normalize(xs)
        center_seleced = self.center.index_select(dim=0, index=ys.long())
        cls_sum_count = ys.float().cpu().histc(bins=self.cls_num, min=0, max=self.cls_num - 1).cuda()
        cls_count_dis = cls_sum_count.index_select(dim=0, index=ys.long())
        return torch.sum(torch.sqrt(torch.sum((xs - center_seleced) ** 2, dim=1)) / cls_count_dis)


class ClsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.PReLU(),
                                        nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.PReLU(),
                                        nn.MaxPool2d(3, 2))
        self.feature_layer = nn.Sequential(nn.Linear(11*11*64, 256), nn.BatchNorm1d(256), nn.PReLU(),
                                           nn.Linear(256, 128), nn.BatchNorm1d(128), nn.PReLU(),
                                           nn.Linear(128, 2), nn.PReLU())
        self.out_layer = nn.Sequential(nn.Linear(2, 10))
        self.loss_fn1 = CenterLoss(2, 10)
        self.loss_fn2 = nn.CrossEntropyLoss()

    def forward(self, x):
        conv = self.conv_layer(x)
        conv = conv.reshape(x.size(0), -1)
        self.feature = self.feature_layer(conv)
        self.out = self.out_layer(self.feature)
        return self.feature

    def get_loss(self, ys, alpha):
        loss1 = self.loss_fn1(self.feature, ys)
        loss2 = self.loss_fn2(self.out, ys.long())
        return alpha * loss1 + (1 - alpha) * loss2


if __name__ == '__main__':
    weight_save_path = r"./params/center_loss.pt"
    dataset = datasets.MNIST(root="../MyTest01/minist_torch/", train=True, transform=transforms.ToTensor(),
                             download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=512, shuffle=True, num_workers=5)
    cls_net = ClsNet()
    cls_net.cuda()
    if os.path.exists(weight_save_path):
        cls_net.load_state_dict(torch.load(weight_save_path))
    fig, ax = plt.subplots()
    for epoch in range(100000):
        for i, (xs, ys) in enumerate(dataloader):
            # xs = xs.reshape(xs.size(0), -1)
            xs = xs.cuda()
            ys = ys.cuda()
            coordinate = cls_net(xs)
            coordinate = coordinate.cpu().detach().numpy()
            loss = cls_net.get_loss(ys, 0.5)
            optimizer = optim.Adam(cls_net.parameters())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.cpu().detach().item())
            torch.save(cls_net.state_dict(), weight_save_path)
            print("save success")
            ys = ys.cpu().numpy()
            plt.cla()
            plt.ion()
            print(np.unique(ys))
            for j in np.unique(ys):
                xx = coordinate[ys == j][:, 0]
                yy = coordinate[ys == j][:, 1]
                ax.scatter(xx, yy)
            # if i % 1000 == 0:
            #     fig.savefig(r"D:/center_loss.jpg")
            plt.show()
            plt.pause(0.1)
            plt.ioff()
