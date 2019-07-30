import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
import utils
import MTCNN_Test as MTCNNnet
from torchvision import transforms
import time
import os


class Detector:
    def __init__(self, pnet_param=r"./param_widerface/test_weight1/pnet.pt",
                 rnet_param="./param_widerface/test_weight1/rnet.pt",
                 onet_param="./param_widerface/test_weight1/onet.pt",
                 isCuda=True):
        self.isCuda = isCuda
        self.pnet = MTCNNnet.PNet()
        self.rnet = MTCNNnet.RNet()
        self.onet = MTCNNnet.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        # load = torch.load(onet_param)
        # print([k for k, v in load.items()])
        # exit()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        # self.__image_transform = transforms.Compose(
        #     [transforms.ToTensor()])
        self.__image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

    def detect(self, image):
        start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return pnet_boxes, np.array([]), np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return pnet_boxes, rnet_boxes, np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return pnet_boxes, rnet_boxes, onet_boxes

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

        return utils.nms(np.array(boxes), 0.5)

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

        # idxs, _ = np.where(cls > 0.9)
        idxs = torch.nonzero(cls > 0.9)[:, 0]

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

        return utils.nms(np.array(boxes), 0.4)

    def __onet_detect(self, image, rnet_boxes):
        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            # img_data = torch.Tensor(np.array(img) / 255. - 0.5)
            # img_data = img_data.permute(2, 0, 1)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)
        cls = _cls.cpu().data
        offset = _offset.cpu().data

        # for full convolution
        cls = cls.view(-1, 1)
        offset = offset.view(-1, 4)

        idxs = torch.nonzero(cls > 0.99999)[:, 0]

        _boxs = torch.tensor(rnet_boxes)[idxs]
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

        boxes = torch.stack((x1, y1, x2, y2, cls[idxs][:, 0]), dim=1)
        # print(len(boxes))
        return utils.nms(np.array(boxes), 0.2, isMin=True)

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
    if 0:
        anno_src = r"D:\widerface\wider_face_split\wider_face_test_filelist.txt"
        img_dir = r"D:\widerface\WIDER_test\images"
        for i, line in enumerate(open(anno_src)):
            image_filename = line.strip()
            image_file = os.path.join(img_dir, image_filename)
            detector = Detector()
            with Image.open(image_file) as img:
                p_boxes, r_boxes, o_boxes = detector.detect(img)
                print(img.size)
                for boxes_type in [o_boxes]:
                    img = Image.open(image_file)
                    imgDraw = ImageDraw.Draw(img)
                    for box in boxes_type:
                        x1 = int(box[0])
                        y1 = int(box[1])
                        x2 = int(box[2])
                        y2 = int(box[3])

                        # print(box[4])
                        imgDraw.rectangle((x1, y1, x2, y2), outline='red')

                    img.show()
    else:
        img_dir = r"C:\test_image"
        for i, line in enumerate(os.listdir(img_dir)):
            image_filename = line.strip()
            image_file = os.path.join(img_dir, image_filename)
            detector = Detector()

            with Image.open(image_file) as img:
                p_boxes, r_boxes, o_boxes = detector.detect(img)
                print(img.size)
                for boxes_type in [o_boxes]:
                    img = Image.open(image_file)
                    imgDraw = ImageDraw.Draw(img)
                    for box in boxes_type:
                        x1 = int(box[0])
                        y1 = int(box[1])
                        x2 = int(box[2])
                        y2 = int(box[3])

                        # print(box[4])
                        imgDraw.rectangle((x1, y1, x2, y2), outline='red')

                    img.show()