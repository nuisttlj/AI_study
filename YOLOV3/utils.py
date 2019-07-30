import torch
import numpy as np


def pic_pad_to_square(img_data):
    if img_data.size(1) > img_data.size(2):
        fill_len1 = (img_data.size(1) - img_data.size(2)) // 2
        fill_len2 = img_data.size(1) - img_data.size(2) - fill_len1
        img_data = torch.cat((torch.zeros((img_data.size(0), img_data.size(1), fill_len1)), img_data,
                              torch.zeros((img_data.size(0), img_data.size(1), fill_len2))), dim=2)
    elif img_data.size(1) < img_data.size(2):
        fill_len1 = (img_data.size(2) - img_data.size(1)) // 2
        fill_len2 = img_data.size(2) - img_data.size(1) - fill_len1
        img_data = torch.cat((torch.zeros((img_data.size(0), fill_len1, img_data.size(2))), img_data,
                              torch.zeros((img_data.size(0), fill_len2, img_data.size(2)))), dim=1)

    return img_data


def iou(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(xx2 - xx1, 0)
    h = np.maximum(yy2 - yy1, 0)

    intersection = w * h
    if isMin:
        over = np.true_divide(intersection, boxes_area)
    else:
        over = np.true_divide(intersection, (box_area + boxes_area - intersection))

    return over


def nms(boxes, threshold=0.3, isMin=False):
    remaind_boxes = []
    if boxes.shape[0] > 0:
        _boxes = boxes[(-boxes[:, 4]).argsort()]
        while _boxes.shape[0] > 1:
            a_box = _boxes[0]
            b_box = _boxes[1:]
            remaind_boxes.append(a_box)
            index = np.where(iou(a_box, b_box, isMin) < threshold)
            _boxes = b_box[index]

        if _boxes.shape[0] > 0:
            remaind_boxes.append(_boxes[0])
        remaind_boxes = np.stack(remaind_boxes)
    else:
        remaind_boxes = np.array(remaind_boxes)
    return remaind_boxes
