import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from model_nn_simple import Nest_Net
import torch.optim as optim
from Dataset import TumorDataSet
import shutil


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)

        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class Dice_coeff(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()

        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)


class Trainer(nn.Module):
    def __init__(self, save_path, use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda
        self.sava_path = save_path
        self.bceloss = nn.BCELoss()
        self.diceloss = DiceLoss()
        self.net = Nest_Net(1, deep_supervision=True)


    def forward(self, x, y):
        if self.use_cuda:
            if torch.cuda.is_available():
                self.net(x).cuda()
                x = x.cuda()
                y = y.cuda()
        self.net(x).apply(weight_init)
        if os.path.exists(self.save_path):
            self.net(x).load_state_dict(torch.load(self.save_path))

        out1, out2, out3, out4 = self.net(x)
        bceloss = (self.bceloss(out1, y) + self.bceloss(out2, y) + self.bceloss(out3, y) + self.bceloss(out4, y)) / 4
        diceloss = (self.diceloss(out1, y) + self.diceloss(out2, y) + self.diceloss(out3, y) + self.diceloss(out4,
                                                                                                             y)) / 4
        # print(bceloss)
        # print(diceloss)
        total_loss = bceloss + diceloss

        optimizer = optim.Adam(self.net(x).parameters())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.cpu().detach().item()


if __name__ == '__main__':
    tumor_data_set = TumorDataSet()
    tumor_data_loader = DataLoader(tumor_data_set, batch_size=2, num_workers=1, shuffle=True, drop_last=True)
    save_path = r"./params/unet++.pt"
    trainer = Trainer(save_path)

    for epoch in range(100000):
        for i, (xs, ys) in enumerate(tumor_data_loader):
            loss = trainer(xs, ys)
            print(loss)

        torch.save(trainer.net.state_dict(), save_path)
        shutil.copy(save_path, save_path + "_bak")
        print("save success")
