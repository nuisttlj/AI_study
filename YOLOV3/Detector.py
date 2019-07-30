import numpy as np
from PIL import Image, ImageDraw
from YOLOV3 import YoloV3Net_simple
from torchvision import transforms
import os
import torch
from YOLOV3 import Cfg as cfg
import matplotlib.pyplot as plt
import PIL.ImageFont as font
from YOLOV3 import utils

LABEL_FILE_PATH = r'/home/coco/labels/train2014test500.txt'
IMAGE_BASE_DIR = r'/home/coco/train2014'
weight_save_path = r'./params/yolov3_500_paper.pt'
my_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])
net = YoloV3Net_simple.YoloV3_Net()
net.cuda()
if os.path.exists(weight_save_path):
    net.load_state_dict(torch.load(weight_save_path))
    print("load success")
else:
    exit()


def back_to_box(label, scale):
    label = label[0]
    label = torch.sigmoid(label)
    label = label.permute(1, 2, 0)
    label = label.reshape(label.shape[0], label.shape[1], 3, -1)
    # label[..., 0] = torch.sigmoid(label[..., 0])
    label = label.cpu().detach().numpy()
    obj_mask = label[..., 0] > 0.5
    obj_content = label[obj_mask]
    indexs = np.nonzero(obj_mask)

    conf = obj_content[:, 0]
    cls = np.argmax(obj_content[:, 5:], axis=1)

    cx = (indexs[1] + obj_content[:, 1]) * (cfg.IMG_WIDTH / scale)
    cy = (indexs[0] + obj_content[:, 2]) * (cfg.IMG_HEIGHT / scale)

    # w = (np.exp(obj_content[:, 3]) - 1) * np.array(cfg.ANCHORS_GROUP[scale])[indexs[2], 0]
    # h = (np.exp(obj_content[:, 4]) - 1) * np.array(cfg.ANCHORS_GROUP[scale])[indexs[2], 1]

    w = (obj_content[:, 3] * 2) ** 2 * np.array(cfg.ANCHORS_GROUP[scale])[indexs[2], 0]
    h = (obj_content[:, 4] * 2) ** 2 * np.array(cfg.ANCHORS_GROUP[scale])[indexs[2], 1]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # filter = np.stack((indexs * cfg.IMG_WIDTH / scale, conf), axis=1)
    boxes = np.stack((x1, y1, x2, y2, conf, cls), axis=1)

    return boxes


with open(LABEL_FILE_PATH) as f:
    my_font = font.truetype(r"./msyh.ttf", size=15)
    cls_name = open("./coco.names", "r").read().splitlines()
    lines = f.readlines()
    for i, line in enumerate(lines):
        img = Image.open(os.path.join(IMAGE_BASE_DIR, line.split()[0]))
        img_width, img_height = img.size
        # scale = max(img_width, img_height) / 416
        new_img_width, new_img_height = img_width / max(img_width, img_height) * 416, img_height / max(img_width,
                                                                                                       img_height) * 416
        img = img.resize((int(new_img_width), int(new_img_height)))
        img_data = my_transforms(img)
        img_data = utils.pic_pad_to_square(img_data)
        img = np.array(img_data.permute(1, 2, 0)) + 0.5
        img = Image.fromarray(np.uint8(img * 255))
        img_data = img_data.unsqueeze(0)
        img_data = img_data.cuda()
        label1, label2, label3 = net(img_data)
        boxes1 = back_to_box(label1, 13)
        boxes2 = back_to_box(label2, 26)
        boxes3 = back_to_box(label3, 52)

        boxes = np.concatenate((boxes1, boxes2, boxes3), axis=0)
        boxes_list = []

        for cls in np.unique(boxes[:, -1]):
            idxs = boxes[:, -1] == cls
            temp_boxes = boxes[idxs]
            temp_boxes_nms = utils.nms(temp_boxes, 0.5)
            # temp_boxes_nms = utils.nms(temp_boxes_nms, 0.5, isMin=True)
            boxes_list.append(temp_boxes_nms)

        if boxes_list:
            boxes = np.concatenate(boxes_list, axis=0)

            draw = ImageDraw.Draw(img)
            for box in boxes:
                draw.rectangle((int(box[0]), int(box[1]), int(box[2]), int(box[3])), outline="blue")
                draw.text(((int(box[0]) + int(box[2])) / 2, int(box[1])), text=cls_name[int(box[5])], font=my_font,
                          fill="red")
        # img.show()
        img.save("./pic/{0}.jpg".format(i))
        # img_plt = np.array(img)
        # plt.imshow(img_plt)
        # plt.pause(1)
