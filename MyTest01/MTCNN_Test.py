import torch.nn as nn
import torch


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU()
        )
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        cond = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cond, offset


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Dropout(0.1),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.3)
        )
        self.fc4 = nn.Linear(64 * 2 * 2, 128)
        self.prelu4 = nn.PReLU()

        self.fc5_1 = nn.Linear(128, 1)
        self.fc5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc4(x)
        x = self.prelu4(x)
        label = torch.sigmoid(self.fc5_1(x))
        offset = self.fc5_2(x)
        return label, offset


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=3),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Dropout(0.4)
        )
        self.conv5_1 = nn.Conv2d(128, 1, kernel_size=1, stride=1)
        self.conv5_2 = nn.Conv2d(128, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        label = torch.sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)
        return label, offset