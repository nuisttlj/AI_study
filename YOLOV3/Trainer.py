from YOLOV3 import DataSet, YoloV3Net_simple
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import shutil


class Trainer:
    def __init__(self, weight_save_path, alpha, isCuda=True):
        self.net = YoloV3Net_simple.YoloV3_Net()
        self.isCuda = isCuda
        self.weight_save_path = weight_save_path
        self.alpha = alpha
        if self.isCuda:
            self.net.cuda()
        self.loss_func0_0 = nn.KLDivLoss()
        self.loss_func0_1 = nn.BCELoss()
        self.loss_func1 = nn.MSELoss()
        self.loss_func2 = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.Adam(self.net.parameters())
        # self.optimizer = optim.RMSprop(self.net.parameters(), lr=1e-4)
        if os.path.exists(self.weight_save_path):
            self.net.load_state_dict(torch.load(self.weight_save_path))

    def compute_loss(self, output, target):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        mask_obj = target[..., 0] > 0
        mask_noobj = target[..., 0] == 0
        # mask_obj_offset = target[..., 0] >= 0.4
        target_conf = target[mask_obj][..., 0].reshape(-1, 3)
        max_conf = torch.argmax(target_conf, dim=1)
        # loss1_obj = self.loss_func1(output[mask_obj], target[mask_obj])
        # loss1_noobj = self.loss_func1(output[mask_noobj], target[mask_noobj])
        loss0_obj = self.loss_func0_0(torch.log(self.sigmoid(output[mask_obj][..., :1])),
                                      target[mask_obj][..., :1]) + self.loss_func0_0(
            torch.log(1 - self.sigmoid(output[mask_obj][..., :1])), (1 - target[mask_obj][..., :1]))
        loss0_noobj = self.loss_func0_1(self.sigmoid(output[mask_noobj][..., :1]), target[mask_noobj][..., :1])
        if torch.max(mask_obj).item():
            loss1_obj = self.loss_func1(
                output[mask_obj][..., 1:5].reshape(-1, 3, 4)[torch.arange(len(max_conf)), max_conf],
                target[mask_obj][..., 1:5].reshape(-1, 3, 4)[torch.arange(len(max_conf)), max_conf])
            # loss1_noobj = self.loss_func1(output[mask_noobj][..., 1:5], target[mask_noobj][..., 1:5])
            loss2_obj = self.loss_func2(
                output[mask_obj][..., 5:].reshape(-1, 3, 80)[torch.arange(len(max_conf)), max_conf],
                target[mask_obj][..., 5:].reshape(-1, 3)[torch.arange(len(max_conf)), max_conf].long())
        else:
            loss1_obj = torch.tensor(0.)
            loss2_obj = torch.tensor(0.)
        # loss2_noobj = self.loss_func2(
        #     output[mask_noobj][..., 5:], target[mask_noobj][..., 5:][:, 0].long())
        loss = self.alpha * (loss0_obj + loss0_noobj) + (1 - self.alpha) * (loss1_obj + loss2_obj)
        # loss = self.alpha * loss1_obj + (1 - self.alpha) * loss1_noobj
        return loss

    def train(self):
        coco_dataset = DataSet.CocoDataSet()
        data_loader = DataLoader(coco_dataset, batch_size=2, shuffle=True, num_workers=2)
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
    train = Trainer("./params/yolov3_6_kl.pt", 0.7)
    train.train()
