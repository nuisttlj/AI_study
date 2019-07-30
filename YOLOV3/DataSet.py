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

my_font = font.truetype(r"./msyh.ttf", size=12)

LABEL_FILE_PATH = r'/home/coco/labels/train2014test6.txt'
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
        scale = max(img_width, img_height) / 416
        new_img_width, new_img_height = np.ceil(img_width / scale), np.ceil(img_height / scale)
        img_data = img_data.resize((int(new_img_width), int(new_img_height)))
        img_data = my_transforms(img_data)
        img_data = utils.pic_pad_to_square(img_data)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros([feature_size, feature_size, 3, 5 + 1])
            # labels[feature_size] = np.zeros([feature_size, feature_size, 3, 5 + cfg.CLASS_NUM])
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

                for j, anchor in enumerate(anchors):
                    pw = anchor[0]
                    ph = anchor[1]
                    tw = np.log(w / pw + 1)
                    th = np.log(h / ph + 1)
                    p_x1 = cx - pw / 2
                    p_y1 = cy - ph / 2
                    p_x2 = cx + pw / 2
                    p_y2 = cy + ph / 2
                    p_box = [[p_x1, p_y1, p_x2, p_y2]]
                    iou = utils.iou(np.array(gt_box), np.array(p_box))[0]
                    # labels[feature_size][int(index_cy), int(index_cx), j] = np.array([iou, offset_cx, offset_cy, tw,
                    #                                                                   th, *one_hot(cfg.CLASS_NUM,
                    #                                                                   int(cls))])
                    labels[feature_size][int(index_cy), int(index_cx), j] = np.array([iou, offset_cx, offset_cy, tw, th,
                                                                                      float(cls)])
        return img_data, labels[13], labels[26], labels[52]


if __name__ == '__main__':
    cocodataset = CocoDataSet()
    for i in range(100):
        try:
            img_data, label1, label2, label3 = cocodataset[i]
            # print(img_data.size())
            pic = np.array(img_data.permute(1, 2, 0)) + 0.5
            obj_mask = label1[..., 0] > 0
            obj_content = label1[obj_mask]
            indexs = np.nonzero(obj_mask)
            cls = obj_content[..., 5]

            cls_name = open("./coco.names", "r").read().splitlines()

            cx = (indexs[1] + obj_content[:, 1]) * 32
            cy = (indexs[0] + obj_content[:, 2]) * 32

            w = (np.exp(obj_content[:, 3]) - 1) * np.array(cfg.ANCHORS_GROUP[13])[indexs[2], 0]
            h = (np.exp(obj_content[:, 4]) - 1) * np.array(cfg.ANCHORS_GROUP[13])[indexs[2], 1]

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            iou = obj_content[:, 0]

            # xx1 = cx - np.tile(np.array(cfg.ANCHORS_GROUP[13])[:, 0] / 2, int(cx.shape[0] / 3))
            # yy1 = cy - np.tile(np.array(cfg.ANCHORS_GROUP[13])[:, 1] / 2, int(cx.shape[0] / 3))
            # xx2 = cx + np.tile(np.array(cfg.ANCHORS_GROUP[13])[:, 0] / 2, int(cx.shape[0] / 3))
            # yy2 = cy + np.tile(np.array(cfg.ANCHORS_GROUP[13])[:, 1] / 2, int(cx.shape[0] / 3))

            boxes = np.stack((x1, y1, x2, y2, iou, cls), axis=1)
            # gt_boxes = np.stack((xx1, yy1, xx2, yy2), axis=1)
            img = Image.fromarray(np.uint8(pic * 255))
            draw = ImageDraw.Draw(img)
            for k, box in enumerate(boxes):
                draw.rectangle((int(box[0]), int(box[1]), int(box[2]), int(box[3])), outline="blue")
                # draw.rectangle((int(gt_boxes[k][0]), int(gt_boxes[k][1]), int(gt_boxes[k][2]), int(gt_boxes[k][3])),
                #                outline="red")
                # draw.text(((int(gt_boxes[k][0]) + int(gt_boxes[k][2])) / 2, int(gt_boxes[k][1])),
                #           text=str(round(box[4], 2)), font=my_font,
                #           fill="blue")
                draw.text(((int(box[0]) + int(box[2])) / 2, int(box[1])), text=cls_name[int(box[5])], fill="red")
            img.show()

        except Exception as e:
            traceback.print_exc()
            exit()
