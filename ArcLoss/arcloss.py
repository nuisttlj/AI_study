import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from lookahead import Lookahead


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)).cuda())
        self.func = nn.Softmax()

    def forward(self, x, s, m):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)
        # 将cosa变小，防止acosa梯度爆炸
        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)
        # 这里再乘回来
        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))

        # 这里arcsomax的概率和不为1，这会导致交叉熵损失看起来很大，且最优点损失也很大
        # print(torch.sum(arcsoftmax, dim=1))
        arcsoftmax = arcsoftmax * (1/torch.sum(arcsoftmax, dim=1, keepdim=True))
        # print(torch.sum(arcsoftmax, dim=1))
        # exit()
        # lmsoftmax = (torch.exp(cosa) - m) / (
        #         torch.sum(torch.exp(cosa) - m, dim=1, keepdim=True) - (torch.exp(cosa) - m) + (torch.exp(cosa) - m))

        return arcsoftmax
        # return lmsoftmax
        # return self.func(torch.matmul(x_norm, w_norm))


class ClsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.PReLU(),
                                        nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.PReLU(),
                                        nn.MaxPool2d(3, 2))
        self.feature_layer = nn.Sequential(nn.Linear(11 * 11 * 64, 256), nn.BatchNorm1d(256), nn.PReLU(),
                                           nn.Linear(256, 128), nn.BatchNorm1d(128), nn.PReLU(),
                                           nn.Linear(128, 2), nn.PReLU())
        self.arcsoftmax = Arcsoftmax(2, 10)
        self.loss_fn = nn.NLLLoss()

    def forward(self, x, s, m):
        conv = self.conv_layer(x)
        conv = conv.reshape(x.size(0), -1)
        feature = self.feature_layer(conv)
        out = self.arcsoftmax(feature, s, m)
        out = torch.log(out)
        return feature, out

    def get_loss(self, out, ys):
        return self.loss_fn(out, ys)
        # print(ys*torch.sum(-out, dim=1))
        # exit()
        # return torch.sum(-ys*out)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)


if __name__ == '__main__':
    weight_save_path = r"./params/arc_loss_test.pt"
    dataset = datasets.MNIST(root="../MyTest01/minist_torch/", train=True, transform=transforms.ToTensor(),
                             download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=True, num_workers=5)
    cls_net = ClsNet()
    cls_net.cuda()
    if os.path.exists(weight_save_path):
        cls_net.load_state_dict(torch.load(weight_save_path))
    fig, ax = plt.subplots()
    optimizer = optim.Adam(cls_net.parameters())
    lookahead = Lookahead(optimizer)
    for epoch in range(100000):
        for i, (xs, ys) in enumerate(dataloader):
            xs = xs.cuda()
            ys = ys.cuda()
            coordinate, out = cls_net(xs, 1, 1)
            coordinate = coordinate.cpu().detach().numpy()
            loss = cls_net.get_loss(out, ys)
            # print([i for i, c in cls_net.named_parameters()])
            # exit()

            lookahead.zero_grad()
            loss.backward()
            lookahead.step()

            print(loss.cpu().detach().item())
            torch.save(cls_net.state_dict(), weight_save_path)
            print("save success")
            ys = ys.cpu().numpy()
            plt.cla()
            plt.ion()
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            coordinate = (coordinate / np.abs(coordinate).max()) * 10
            for j in np.unique(ys):
                xx = coordinate[ys == j][:, 0]
                yy = coordinate[ys == j][:, 1]
                ax.scatter(xx, yy)
            # if i % 1000 == 0:
            #     fig.savefig(r"D:/center_loss.jpg")
            plt.show()
            plt.pause(0.1)
            plt.ioff()
