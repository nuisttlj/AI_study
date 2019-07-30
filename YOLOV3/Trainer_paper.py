from YOLOV3 import DataSet_paper, YoloV3Net_simple
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import shutil


class Trainer:
    def __init__(self, weight_save_path, alpha1, alpha2, alpha3, isCuda=True):
        self.net = YoloV3Net_simple.YoloV3_Net()
        self.isCuda = isCuda
        self.weight_save_path = weight_save_path
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        if self.isCuda:
            self.net.cuda()
        self.loss_func1 = nn.BCELoss(reduction='sum')
        self.loss_func2 = nn.MSELoss(reduction='sum')
        self.sigmoid = nn.Sigmoid()
        # self.optimizer = optim.Adam(self.net.parameters())
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=1e-4)
        if os.path.exists(self.weight_save_path):
            self.net.load_state_dict(torch.load(self.weight_save_path))

    def compute_loss(self, output, target):
        output = self.sigmoid(output)
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        mask_obj = target[..., 0] == 1
        mask_noobj = target[..., 0] == 0

        loss0_noobj = self.loss_func1(output[mask_noobj][..., :1], target[mask_noobj][..., :1])
        if torch.max(mask_obj).item():
            loss0_obj = self.loss_func1(output[mask_obj][..., :1], target[mask_obj][..., :1])
            loss1_obj = self.loss_func2(
                output[mask_obj][..., 1:5], target[mask_obj][..., 1:5])
            loss2_obj = self.loss_func1(
                output[mask_obj][..., 5:], target[mask_obj][..., 5:]) / 80

        else:
            loss0_obj = torch.tensor(0.).cuda()
            loss1_obj = torch.tensor(0.).cuda()
            loss2_obj = torch.tensor(0.).cuda()

        loss = self.alpha1 * (loss0_obj + loss2_obj) + self.alpha2 * loss1_obj + self.alpha3 * loss0_noobj

        return loss

    def train(self):
        coco_dataset = DataSet_paper.CocoDataSet()
        data_loader = DataLoader(coco_dataset, batch_size=5, shuffle=True, num_workers=5, drop_last=True)
        for epoch in range(10000):
            for i, (img_data, target13, target26, target52) in enumerate(data_loader):
                self.net.train()
                if self.isCuda:
                    img_data = img_data.cuda()
                    target13 = target13.float().cuda()
                    target26 = target26.float().cuda()
                    target52 = target52.float().cuda()

                output13, output26, output52 = self.net(img_data)

                loss13 = self.compute_loss(output13, target13)
                loss26 = self.compute_loss(output26, target26)
                loss52 = self.compute_loss(output52, target52)

                loss = loss13 + loss26 + loss52

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("epoch ", epoch, " : ", loss.cpu().detach().item())
            torch.save(self.net.state_dict(), self.weight_save_path)
            print("save success")
            if epoch % 10 == 0:
                shutil.copy(self.weight_save_path, "{0}_bak".format(self.weight_save_path))


if __name__ == '__main__':
    train = Trainer("./params/yolov3_500_paper.pt", 1, 2, 0.1)
    train.train()
