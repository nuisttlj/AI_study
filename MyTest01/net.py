import torch
import torch.nn as nn
import torch.nn.functional as F


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.per_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.PReLU()
        )

        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.per_layer(x)
        cond = F.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cond, offset


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1),
            nn.PReLU()
        )

        self.linear4 = nn.Linear(64 * 2 * 2, 128)
        self.prelu4 = nn.PReLU()

        self.linear5_1 = nn.Linear(128, 1)
        self.linear5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear4(x)
        x = self.prelu4(x)
        cond = F.sigmoid(self.linear5_1(x))
        offset = self.linear5_2(x)
        return cond, offset


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.PReLU()
        )

        self.linear5 = nn.Linear(128 * 2 * 2, 256)
        self.prelu5 = nn.PReLU()

        self.linear6_1 = nn.Linear(256, 1)
        self.linear6_2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear5(x)
        x = self.prelu5(x)
        cond = F.sigmoid(self.linear6_1(x))
        offset = self.linear6_2(x)
        return cond, offset
