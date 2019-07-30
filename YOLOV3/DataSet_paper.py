from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import math
from PIL import Image, ImageDraw
import os
from YOLOV3 import Cfg as cfg
import PIL.ImageFont as font
import traceback
from YOLOV3 import utils
import matplotlib.pyplot as plt

my_font = font.truetype(r"./msyh.ttf", size=15)

LABEL_FILE_PATH = r'/home/coco/labels/train2014test500.txt'
IMAGE_BASE_DIR = r'/home/coco/train2014'

my_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])


def one_hot(class_num, v):
    vector = np.zeros(class_num)
    vector[v] = 1.
    return vector


class CocoDataSet(Dataset):
    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}
        line = self.dataset[index]
        strs = line.split()
        img_data = Image.open(os.path.join(IMAGE_BASE_DIR, strs[0]))
        img_width, img_height = img_data.size
        # scale = max(img_width, img_height) / 416
        new_img_width, new_img_height = img_width / max(img_width, img_height) * 416, img_height / max(img_width,
                                                                                                       img_height) * 416
        if max(new_img_width, new_img_height) != 416:
            print("error")
        img_data = img_data.resize((int(new_img_width), int(new_img_height)))
        img_data = my_transforms(img_data)
        img_data = utils.pic_pad_to_square(img_data)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros([feature_size, feature_size, 3, 5 + cfg.CLASS_NUM])
            boxes = strs[1:]
            boxes = np.split(np.array(boxes, dtype=np.float32), len(boxes) // 5)

            for box in boxes:
                cls, cx, cy, w, h = box
                cx = cx * new_img_width + (416 - new_img_width) / 2
                cy = cy * new_img_height + (416 - new_img_height) / 2
                w = w * new_img_width
                h = h * new_img_height
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                gt_box = [x1, y1, x2, y2]
                offset_cx, index_cx = math.modf(
                    cx / (cfg.IMG_WIDTH / feature_size))
                offset_cy, index_cy = math.modf(
                    cy / (cfg.IMG_HEIGHT / feature_size))

                pw = np.array(anchors)[:, 0]
                ph = np.array(anchors)[:, 1]
                # tw = np.log(w / pw + 1)
                # th = np.log(h / ph + 1)
                tw = (w / pw) ** (1 / 2) / 2
                th = (h / ph) ** (1 / 2) / 2
                p_x1 = cx - pw / 2
                p_y1 = cy - ph / 2
                p_x2 = cx + pw / 2
                p_y2 = cy + ph / 2
                p_box = np.stack((p_x1, p_y1, p_x2, p_y2), axis=1)
                iou = utils.iou(np.array(gt_box), p_box)
                iou = np.where(iou <= 0.5, np.zeros(iou.shape), iou)
                if np.max(iou) > 0.5:
                    iou[np.argmax(iou)] = 1.
                for j in range(len(anchors)):
                    if labels[feature_size][int(index_cy), int(index_cx), j][0] < iou[j]:
                        labels[feature_size][int(index_cy), int(index_cx), j] = np.array(
                            [iou[j], offset_cx, offset_cy, tw[j], th[j], *one_hot(cfg.CLASS_NUM, int(cls))])
        return img_data, labels[13], labels[26], labels[52]


def back_to_box(label, scale):
    obj_mask = label[..., 0] > 0.5
    obj_content = label[obj_mask]
    indexs = np.nonzero(obj_mask)

    conf = obj_content[:, 0]
    cls = np.argmax(obj_content[:, 5:], axis=1)

    cx = (indexs[1] + obj_content[:, 1]) * (cfg.IMG_WIDTH / scale)
    cy = (indexs[0] + obj_content[:, 2]) * (cfg.IMG_HEIGHT / scale)

    w = (obj_content[:, 3] * 2) ** 2 * np.array(cfg.ANCHORS_GROUP[scale])[indexs[2], 0]
    h = (obj_content[:, 4] * 2) ** 2 * np.array(cfg.ANCHORS_GROUP[scale])[indexs[2], 1]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # filter = np.stack((indexs * cfg.IMG_WIDTH / scale, conf), axis=1)
    boxes = np.stack((x1, y1, x2, y2, conf, cls), axis=1)

    return boxes


if __name__ == '__main__':
    cocodataset = CocoDataSet()
    for i in range(100):
        try:
            img_data, label1, label2, label3 = cocodataset[i]
            # print(img_data.size())
            pic = np.array(img_data.permute(1, 2, 0)) + 0.5

            cls_name = open("./coco.names", "r").read().splitlines()

            boxes1 = back_to_box(label1, 13)
            boxes2 = back_to_box(label2, 26)
            boxes3 = back_to_box(label3, 52)

            boxes = np.concatenate((boxes1, boxes2, boxes3), axis=0)
            img = Image.fromarray(np.uint8(pic * 255))
            draw = ImageDraw.Draw(img)
            for k, box in enumerate(boxes):
                draw.rectangle((int(box[0]), int(box[1]), int(box[2]), int(box[3])), outline="blue")
                draw.text(((int(box[0]) + int(box[2])) / 2, int(box[1])), text=cls_name[int(box[5])], fill="red")
            # img.show()
            img_plt = np.array(img)
            plt.imshow(img_plt)
            plt.pause(1)
        except Exception as e:
            # traceback.print_exc()
            exit()
