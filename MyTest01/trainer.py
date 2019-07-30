import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from sampling import FaceDataset
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import offset_iou
import numpy as np
import time
import shutil


class Trainer:
    def __init__(self, net, save_path, train_set_path, dev_set_path, train_pic_size, accuracy_metric_list, figure_name,
                 isCuda=True):
        self.net = net
        self.save_path = save_path
        self.train_set_path = train_set_path
        self.dev_set_path = dev_set_path
        self.train_pic_size = train_pic_size
        self.accuracy_metric_list = accuracy_metric_list
        self.figure_name = figure_name
        self.isCuda = isCuda
        self.continue_train_mask = 0

        if self.isCuda:
            self.net.cuda()

        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())

        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
            self.continue_train_mask = 1

    # 计算ap
    def voc_ap(self, rec, prec, use_07_metric=False):  # voc2007的计算方式和voc2012的计算方式不同，目前一般采用第二种
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def train(self):
        facetrainDataset = FaceDataset(self.train_set_path)
        train_dataloader = DataLoader(facetrainDataset, batch_size=1024, shuffle=True, num_workers=5)
        facedevDataset = FaceDataset(self.dev_set_path)
        dev_dataloader = DataLoader(facedevDataset, batch_size=256, shuffle=True, num_workers=2)
        # plt.ion()
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        yy1 = []
        yy2 = []

        j = 0
        trigger_val = 0
        time_lable = time.strftime('%Y%m%d%H%M%S')
        title1 = "Train recall:  "
        title2 = "Train precision:  "
        title3 = "Dev recall:  "
        title4 = "Dev precision:  "

        for epoch in range(1000):
            for i, (img_data_, category_, offset_, used_label_) in enumerate(train_dataloader):
                # print(img_data_.size())
                self.net.train()
                if self.isCuda:
                    img_data_ = img_data_.cuda()
                    category_ = category_.cuda()
                    offset_ = offset_.cuda()

                _output_category, _output_offset = self.net(img_data_)
                output_category = _output_category.view(-1, 1)
                output_offset = _output_offset.view(-1, 4)

                category_mask = torch.lt(used_label_, 2)
                category = category_[category_mask]
                output_category = output_category[category_mask]
                cls_loss = self.cls_loss_fn(output_category, category)

                offset_mask = torch.gt(used_label_, 0)
                offset_mask = offset_mask.reshape((-1,))
                offset = offset_[offset_mask]
                output_offset = output_offset[offset_mask]
                offset_loss = self.offset_loss_fn(output_offset, offset)
                loss = cls_loss + offset_loss
                loss_train = cls_loss + offset_loss / 2

                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

                print("epoch", epoch + 1, "iterations", i + 1)
                print("loss:", loss.cpu().detach().numpy(), " cls_loss:", cls_loss.cpu().detach().numpy(),
                      " offset_loss:",
                      offset_loss.cpu().detach().numpy())
                if loss < 0.15:
                    trigger_val += 1

                # if j % 1000 == 0:
            title1 = "Train recall:  "
            title2 = "Train:  "
            output_category_sort_index = np.argsort(-output_category.cpu().detach().numpy())
            output_category_sort = np.sort(output_category.cpu().detach().numpy())[::-1]
            category_sort = category.cpu().numpy()[output_category_sort_index]
            # category_mask_accu = torch.gt(category_sort, 0)
            positive_num = category[torch.gt(category, 0)].size(0)
            for n, accuracy_metric in enumerate(self.accuracy_metric_list):
                tp = category_sort[np.where(output_category_sort > accuracy_metric)]
                train_conf_recall = np.cumsum(tp) / positive_num
                # print("conf_accuracy with metric {0} ==> ".format(accuracy_metric), conf_accuracy)
                # title1 += "conf{0}=>{1}; ".format(accuracy_metric, np.round(train_conf_recall, 3))

                temp_count = np.ones(shape=tp.shape)
                train_conf_precision = np.cumsum(tp) / np.cumsum(temp_count)
                train_ap = self.voc_ap(train_conf_recall, train_conf_precision)
                yy1.append(train_ap)
                # print("conf_accuracy with metric {0} ==> ".format(accuracy_metric), conf_accuracy)
                # title2 += "conf{0}=>{1}; ".format(accuracy_metric, np.round(train_conf_precision, 3))
            offset_accuracy = np.mean(
                offset_iou(np.array(output_offset.cpu().detach()), np.array(offset.cpu()), self.train_pic_size))
            # print("offset_accuracy ==> ", offset_accuracy)
            title2 += "offset=>{0}; ".format(np.round(offset_accuracy, 3))
            dev_loss = 0

                    # if (trigger_val >= 100 or self.continue_train_mask) and j % 1000 == 0:
            self.net.eval()
            title3 = "Dev recall:  "
            title4 = "Dev:  "
            with torch.no_grad():
                for k, (dev_img_data, dev_category, dev_offset, dev_used_label) in enumerate(
                        dev_dataloader):
                    # print(img_data_.size())
                    if k == 0:
                        if self.isCuda:
                            dev_img_data = dev_img_data.cuda()
                            dev_category = dev_category.cuda()
                            dev_offset = dev_offset.cuda()

                        _output_category, _output_offset = self.net(dev_img_data)
                        output_category = _output_category.view(-1, 1)
                        output_offset = _output_offset.view(-1, 4)

                        category_mask = torch.lt(dev_used_label, 2)
                        category = dev_category[category_mask]
                        output_category = output_category[category_mask]
                        dev_cls_loss = self.cls_loss_fn(output_category, category)

                        offset_mask = torch.gt(dev_used_label, 0)
                        offset_mask = offset_mask.reshape((-1,))
                        offset = dev_offset[offset_mask]
                        output_offset = output_offset[offset_mask]
                        dev_offset_loss = self.offset_loss_fn(output_offset, offset)

                        dev_loss = dev_cls_loss + dev_offset_loss

                        print("dev_set for cross validation")
                        print("dev_loss:", dev_loss.cpu().detach().numpy(), " dev_cls_loss:",
                              dev_cls_loss.cpu().detach().numpy(), " dev_offset_loss:",
                              dev_offset_loss.cpu().detach().numpy())

                        output_category_sort_index = np.argsort(-output_category.cpu().detach().numpy())
                        output_category_sort = np.sort(output_category.cpu().detach().numpy())[::-1]
                        category_sort = category.cpu().numpy()[output_category_sort_index]
                        # category_mask_accu = torch.gt(category_sort, 0)
                        positive_num = category[torch.gt(category, 0)].size(0)

                        for n, accuracy_metric in enumerate(self.accuracy_metric_list):
                            # conf_recall = np.mean(
                            #     np.equal(
                            #         np.where(output_category[
                            #                      category_mask_accu].cpu().detach().numpy() > accuracy_metric,
                            #                  1,
                            #                  0),
                            #         category[category_mask_accu].cpu().numpy()))
                            tp = category_sort[np.where(output_category_sort > accuracy_metric)]
                            dev_conf_recall = np.cumsum(tp) / positive_num
                            # title3 += "conf{0}=>{1}; ".format(accuracy_metric, np.round(dev_conf_recall, 3))

                            # conf_accuracy = np.mean(
                            #     np.equal(
                            #         np.where(output_category.cpu().detach().numpy() > accuracy_metric, 1,
                            #                  0),
                            #         category.cpu().numpy()))
                            # conf_precision = np.mean(
                            #     category.cpu().numpy()[
                            #         np.where(output_category.cpu().detach().numpy() > accuracy_metric)])

                            temp_count = np.ones(shape=tp.shape)
                            dev_conf_precision = np.cumsum(tp) / np.cumsum(temp_count)
                            dev_ap = self.voc_ap(dev_conf_recall, dev_conf_precision)
                            yy2.append(dev_ap)
                            # title4 += "conf{0}=>{1}; ".format(accuracy_metric,
                                                              # np.round(dev_conf_precision, 3))
                        offset_accuracy = np.mean(
                            offset_iou(np.array(output_offset.cpu().detach()), np.array(offset.cpu()),
                                       self.train_pic_size))
                        # print("offset_accuracy ==> ", offset_accuracy)
                        title4 += "offset=>{0}; ".format(np.round(offset_accuracy, 3))

                        # if j != 0 and j % 1000 == 0:
                        #     f_loss = open(
                        #         r"D:\PycharmProjects\MyTest01\param_widerface_new\{0}_loss.txt".format(
                        #             self.figure_name), "a")
                        #     f_loss.write(
                        #         "epoch:{0}; iterations:{1}; loss: train{2}, dev{5}; cls_loss: train{3}, "
                        #         "dev{6}; offset_loss: train{4}, dev{7};\n train_recall: {8}; "
                        #         "train_accuracy: {9}; dev_recall: {10}; dev_accuracy: {11} \n \n \n".format(
                        #             epoch + 1, j + 1,
                        #             loss.cpu().detach().numpy(),
                        #             cls_loss.cpu().detach().numpy(),
                        #             offset_loss.cpu().detach().numpy(),
                        #             dev_loss.cpu().detach().numpy(),
                        #             dev_cls_loss.cpu().detach().numpy(),
                        #             dev_offset_loss.cpu().detach().numpy(),
                        #             title1, title2, title3, title4))
                        #     f_loss.close()

                    else:
                        break

            torch.save(self.net.state_dict(), self.save_path)
            print("save success")

            # x1.append(j + 1)
            x1.append(epoch+1)
            y1.append(loss.cpu().detach().item())
            if dev_loss:
                # x2.append(j + 1)
                x2.append(epoch + 1)
                y2.append(dev_loss)
            fig = plt.figure(self.figure_name, figsize=(10, 10), dpi=120)
            plt.clf()
            ax1 = plt.subplot(511)
            train, = plt.plot(x1, y1, linewidth=1, color="red")
            dev, = plt.plot(x2, y2, linewidth=1, color="blue")
            plt.legend([train, dev], ["train", "dev"], loc="upper right", fontsize=8)
            # plt.title(title1 + "\n" + title2 + "\n" + title3 + "\n" + title4, fontsize=7)
            plt.title("LOSS:  " + title2 + "\n" + title4, fontsize=12)

            ax2 = plt.subplot(512)
            train, = plt.plot(x1, yy1[0::4], linewidth=1, color="red")
            dev, = plt.plot(x2, yy2[0::4], linewidth=1, color="blue")
            plt.legend([train, dev], ["train", "dev"], loc="upper right", fontsize=8)
            plt.title("Ap: " + str(self.accuracy_metric_list[0]), fontsize=12)

            ax3 = plt.subplot(513)
            train, = plt.plot(x1, yy1[1::4], linewidth=1, color="red")
            dev, = plt.plot(x2, yy2[1::4], linewidth=1, color="blue")
            plt.legend([train, dev], ["train", "dev"], loc="upper right", fontsize=8)
            plt.title("Ap: " + str(self.accuracy_metric_list[1]), fontsize=12)

            ax4 = plt.subplot(514)
            train, = plt.plot(x1, yy1[2::4], linewidth=1, color="red")
            dev, = plt.plot(x2, yy2[2::4], linewidth=1, color="blue")
            plt.legend([train, dev], ["train", "dev"], loc="upper right", fontsize=8)
            plt.title("Ap: " + str(self.accuracy_metric_list[2]), fontsize=12)

            ax5 = plt.subplot(515)
            train, = plt.plot(x1, yy1[3::4], linewidth=1, color="red")
            dev, = plt.plot(x2, yy2[3::4], linewidth=1, color="blue")
            plt.legend([train, dev], ["train", "dev"], loc="upper right", fontsize=8)
            plt.title("Ap: " + str(self.accuracy_metric_list[3]), fontsize=12)

            fig.tight_layout()
            # plt.pause(0.01)
            # if j-1 % 100 == 0:
            fig.savefig(r"D:\PycharmProjects\MyTest01\param_widerface_new/{0}{1}.jpg".format(self.figure_name,
                                                                                             time_lable))

                # j += 1
            if epoch >= 0:
                shutil.copy(self.save_path, "{0}_epoch_{1}".format(self.save_path, epoch + 1))

        # plt.ioff()
