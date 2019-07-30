import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
import utils
# import net as MTCNNnet
import MTCNNnet_samestride as MTCNNnet
from torchvision import transforms
import time
import os


class Detector:
    def __init__(self, pnet_param=r"./param_widerface_new/pnet.pt",
                 rnet_param="./param_widerface_new/rnet.pt",
                 isCuda=True):
        self.isCuda = isCuda
        self.pnet = MTCNNnet.PNet()
        self.rnet = MTCNNnet.RNet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))

        self.pnet.eval()
        self.rnet.eval()

        # self.__image_transform = transforms.Compose(
        #     [transforms.ToTensor()])
        self.__image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

    def detect(self, image):
        # start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
        # end_time = time.time()
        # t_pnet = end_time - start_time

        # start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return pnet_boxes, np.array([]), np.array([])
        # end_time = time.time()
        # t_rnet = end_time - start_time

        # t_sum = t_pnet + t_rnet

        # print("total:{0} pnet:{1} rnet:{2}".format(t_sum, t_pnet, t_rnet))

        return pnet_boxes, rnet_boxes

    def __pnet_detect(self, image):
        img = image
        w, h = img.size
        # img = img.resize((int(w * 0.5), int(h * 0.5)))
        # img_data = image
        # w, h  = img_data.shape[1], img_data.shape[0]
        min_side_len = min(w, h)
        scale = 1.
        total_idxs = []
        total_offset = []
        total_scales = []
        total_cls = []

        while min_side_len > 12:
            img_data = self.__image_transform(img)
            # img_data = torch.Tensor(np.array(img) / 255. - 0.5)
            # img_data = img_data.permute(2, 0, 1)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data = img_data.unsqueeze(0)

            _cls, _offset = self.pnet(img_data)
            cls, offset = _cls[0][0].cpu().detach(), _offset[0].cpu().detach()
            idxs = torch.nonzero(torch.gt(cls, 0.9))
            scales = torch.full((idxs.shape[0],), scale)
            # if self.isCuda:
            #     idxs = idxs.cuda()
            #     scales = scales.cuda()
            offset_selected = offset[:, idxs[:, 0], idxs[:, 1]]
            cls_selected = cls[idxs[:, 0], idxs[:, 1]]

            total_idxs.append(idxs)
            total_offset.append(offset_selected)
            total_scales.append(scales)
            total_cls.append(cls_selected)

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)

        total_idxs = torch.cat(total_idxs, dim=0)
        total_scales = torch.cat(total_scales, dim=0)
        total_offset = torch.cat(total_offset, dim=1)
        total_cls = torch.cat(total_cls, dim=0)

        boxes = self.__box(total_idxs, total_offset, total_cls, total_scales)

        return utils.nms(np.array(boxes), 0.1)

        # return np.array(boxes)

    def __rnet_detect(self, image, pnet_boxes):
        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            # img_data = torch.Tensor(np.array(img) / 255. - 0.5)
            # img_data = img_data.permute(2, 0, 1)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)

        cls = _cls.cpu().data
        offset = _offset.cpu().data

        # for full convolution
        cls = cls.view(-1, 1)
        offset = offset.view(-1, 4)

        # idxs, _ = np.where(cls > 0.9)
        idxs = torch.nonzero(cls > 0.99)[:, 0]
        # idxs = torch.nonzero((cls > 0.90) & (cls < 0.95))[:, 0]

        _boxs = torch.tensor(_pnet_boxes)[idxs]
        # _x1 = np.array(_boxs[:, 0], dtype=np.int32)
        # _y1 = np.array(_boxs[:, 1], dtype=np.int32)
        # _x2 = np.array(_boxs[:, 2], dtype=np.int32)
        # _y2 = np.array(_boxs[:, 3], dtype=np.int32)
        _x1 = _boxs[:, 0]
        _y1 = _boxs[:, 1]
        _x2 = _boxs[:, 2]
        _y2 = _boxs[:, 3]

        ow = _x2 - _x1
        oh = _y2 - _y1
        x1 = _x1 + ow * offset[idxs][:, 0]
        y1 = _y1 + oh * offset[idxs][:, 1]
        x2 = _x2 + ow * offset[idxs][:, 2]
        y2 = _y2 + oh * offset[idxs][:, 3]

        # boxes = np.stack((x1, y1, x2, y2, cls[idxs][:, 0]), axis=1)
        boxes = torch.stack((x1, y1, x2, y2, cls[idxs][:, 0]), dim=1)

        return utils.nms(np.array(boxes), 0.1)
        # return np.array(boxes)

    def __box(self, start_index, offset, cls, scale, stride=torch.tensor(2.), side_len=torch.tensor(12.)):
        # if self.isCuda:
        #     scale = scale.cuda()
        #     stride = stride.cuda()
        #     side_len = side_len.cuda()

        _x1 = (start_index[:, 1].float() * stride) / scale
        _y1 = (start_index[:, 0].float() * stride) / scale
        _x2 = (_x1 + side_len / scale)
        _y2 = (_y1 + side_len / scale)

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[0]
        y1 = _y1 + oh * offset[1]
        x2 = _x2 + ow * offset[2]
        y2 = _y2 + oh * offset[3]

        return torch.stack((x1, y1, x2, y2, cls), dim=1)


if __name__ == '__main__':
    if 1:
        anno_src = r"D:\widerface\wider_face_split\wider_face_val_bbx_gt.txt"
        img_dir = r"D:\widerface\WIDER_val\images"
        negative_anno_filename = r"C:\mywiderface_dev\48\negative1.txt"
        negative_image_dir = r"C:\mywiderface_dev\48\negative1"
        if not os.path.exists(negative_image_dir):
            os.makedirs(negative_image_dir)
        negative_anno_file = open(negative_anno_filename, "a")
        negative_count = 0
        num = 0
        all_strs = open(anno_src).readlines()
        # print(len([i for i in all_strs if ".jpg" in i]))
        # exit()
        for i in range(len(all_strs)):
            if ".jpg" in all_strs[i]:
                # if num >= 5711:
                if num >= 0:
                    image_filename = all_strs[i].strip()
                    image_file = os.path.join(img_dir, image_filename)
                    with Image.open(image_file) as img:
                        boxes = []
                        img_w, img_h = img.size
                        bbox_num = int(all_strs[i + 1])
                        # if bbox_num < 10:
                        for j in range(bbox_num):
                            pic_str = all_strs[i + 1 + j + 1]
                            pic_str_list = pic_str.strip().split()
                            x1 = int(pic_str_list[0].strip())
                            y1 = int(pic_str_list[1].strip())
                            w = int(pic_str_list[2].strip())
                            h = int(pic_str_list[3].strip())
                            x2 = x1 + w
                            y2 = y1 + h

                            if x1 <= 0 or y1 <= 0 or w <= 0 or h <= 0:
                                continue

                            boxes.append([x1, y1, x2, y2])
                        detector = Detector()
                        try:
                            p_boxes, r_boxes = detector.detect(img)

                            iou = utils.bbox_iou(r_boxes[:, :4], np.array(boxes))
                        except:
                            continue
                        idxs = np.where(iou.max(axis=1) <= 0)
                        remained_boxes = r_boxes[:, :4][idxs]

                        # imgDraw = ImageDraw.Draw(img)
                        for box in remained_boxes:
                            x1 = int(box[0])
                            y1 = int(box[1])
                            x2 = int(box[2])
                            y2 = int(box[3])

                            # print(box[4])
                            # imgDraw.rectangle((x1, y1, x2, y2), outline='red')

                            img_crop = img.crop(np.array([x1, y1, x2, y2]))
                            img_resize = img_crop.resize((48, 48))
                            negative_anno_file.write(
                                "negative1/{0}.jpg {1} 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            img_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1
                num += 1
                # print("进度:  ", num, "--", round(num * 100 / 12880, 2), "%", "---", negative_count - 0)
                print("进度:  ", num, "--", round(num * 100 / 3226, 2), "%", "---", negative_count - 0)
                # img.show()

        negative_anno_file.close()
