import numpy as np
import torch
import time


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


def iou_new(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = torch.max(box[0], boxes[:, 0])
    yy1 = torch.max(box[1], boxes[:, 1])
    xx2 = torch.min(box[2], boxes[:, 2])
    yy2 = torch.min(box[3], boxes[:, 3])

    w = torch.clamp(xx2 - xx1, min=0)
    h = torch.clamp(yy2 - yy1, min=0)

    intersection = w * h
    if isMin:
        over = torch.div(intersection, boxes_area)
    else:
        over = torch.div(intersection, (box_area + boxes_area - intersection))

    return over


def offset_iou(offset1, offset2, train_pic_size):
    boxes1 = offset1 * train_pic_size + np.array([0, 0, train_pic_size, train_pic_size])
    boxes2 = offset2 * train_pic_size + np.array([0, 0, train_pic_size, train_pic_size])

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    xx1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
    yy1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
    xx2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
    yy2 = np.minimum(boxes1[:, 3], boxes2[:, 3])

    w = np.maximum(xx2 - xx1, 0)
    h = np.maximum(yy2 - yy1, 0)

    intersection = w * h
    over = np.true_divide(intersection, (boxes1_area + boxes2_area - intersection))
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


def nms_new(boxes, threshold=0.3, isMin=False):
    remaind_boxes = []
    if boxes.shape[0] > 0:
        _boxes = boxes[(-boxes[:, 4]).argsort()]
        while _boxes.shape[0] > 1:
            a_box = _boxes[0]
            b_box = _boxes[1:]
            remaind_boxes.append(a_box)
            index = torch.nonzero(iou_new(a_box, b_box, isMin) < threshold)
            _boxes = b_box[index[:, 0]]

        if _boxes.shape[0] > 0:
            remaind_boxes.append(_boxes[0])
        remaind_boxes = torch.stack(remaind_boxes)
    # else:
    #     remaind_boxes = np.array(remaind_boxes)
    return remaind_boxes


def convert_to_square(bbox):
    square_bbox = bbox.copy()
    # print(type(square_bbox))
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox


def prewritten():
    pass


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


if __name__ == '__main__':
    boxes = np.array(
        [[10, 10, 20, 20, 1], [8, 9, 18, 19, 0.4], [18, 19, 30, 40, 0.8], [2, 4, 6, 8, 0.2], [21, 23, 28, 37, 0.6]])
    print(nms(boxes))
