import cv2
import torch
from PIL import Image
import numpy as np
import utils
# import net as MTCNNnet
import MTCNNnet_samestride as MTCNNnet
from torchvision import transforms
import time


class Detector:
    def __init__(self, pnet_param=r"./param_widerface_new/pnet.pt",
                 rnet_param="./param_widerface_new/rnet.pt",
                 onet_param="./param_widerface_new/onet.pt",
                 isCuda=True):
        self.isCuda = isCuda
        self.pnet = MTCNNnet.PNet()
        self.rnet = MTCNNnet.RNet()
        self.onet = MTCNNnet.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

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
        # start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        # end_time = time.time()
        # t_pnet = end_time - start_time

        # start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        # end_time = time.time()
        # t_rnet = end_time - start_time

        # start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        # end_time = time.time()
        # t_onet = end_time - start_time

        # t_sum = t_pnet + t_rnet + t_onet

        # print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def __pnet_detect(self, image):
        boxes = []
        img = image
        w, h = img.size
        # if h > 500:
        #     img = img.resize((int(w * 0.5), int(h * 0.5)))
        #     w, h = img.size
        min_side_len = min(w, h)
        scale = 1

        while min_side_len > 12:
            img_data = self.__image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data = img_data.unsqueeze(0)
            # print(img_data)
            # print(img_data.shape)
            # exit()

            _cls, _offset = self.pnet(img_data)
            cls, offset = _cls[0][0].cpu().data, _offset[0].cpu().data
            idxs = torch.nonzero(torch.gt(cls, 0.99))
            for idx in idxs:
                boxes.append(self.__box(idx, offset, cls[idx[0], idx[1]], torch.tensor(scale)))

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)
        # print(np.array(boxes).shape)
        # print(utils.nms(np.array(boxes), 0.5).shape)
        # exit()
        return utils.nms(np.array(boxes), 0.5)

    def __rnet_detect(self, image, pnet_boxes):
        boxes = []
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
            _img_dataset.append(img_data)

        # img_dataset = torch.stack(_img_dataset)
        img_dataset = tuple(_img_dataset)
        img_dataset = torch.cat(img_dataset, 2)
        img_dataset = torch.unsqueeze(img_dataset, 0)
        # print(img_dataset.size())
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)
        _cls = _cls.view(1, -1)
        _cls = _cls.permute(1, 0)
        _offset = _offset.view(4, -1)
        _offset = _offset.permute(1, 0)
        cls = _cls.cpu().detach().numpy()
        offset = _offset.cpu().detach().numpy()
        idxs, _ = np.where(cls > 0.999)
        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.4)

    def __onet_detect(self, image, rnet_boxes):
        boxes = []
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
            _img_dataset.append(img_data)

        # img_dataset = torch.stack(_img_dataset)
        img_dataset = tuple(_img_dataset)
        img_dataset = torch.cat(img_dataset, 2)
        img_dataset = torch.unsqueeze(img_dataset, 0)
        # print(img_dataset.size())
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)
        _cls = _cls.view(1, -1)
        _cls = _cls.permute(1, 0)
        _offset = _offset.view(4, -1)
        _offset = _offset.permute(1, 0)
        cls = _cls.cpu().detach().numpy()
        offset = _offset.cpu().detach().numpy()
        idxs, _ = np.where(cls > 0.9999)
        for idx in idxs:
            _box = rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])
        # print(len(boxes))
        return utils.nms(np.array(boxes), 0.2, isMin=True)

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        _x1 = (start_index[1] * stride) / scale
        _y1 = (start_index[0] * stride) / scale
        _x2 = (_x1 + side_len / scale)
        _y2 = (_y1 + side_len / scale)

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]


if __name__ == '__main__':
    detector = Detector()
    cap = cv2.VideoCapture(r"C:\video\1.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_writer = cv2.VideoWriter(r"C:\video\widerface_out1.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    i = 0
    boxes = []
    boxes_copy = []
    x1 = 0
    non_detected = 0
    start_time = time.time()
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv2.waitKey(1000)
            cap.release()
            vid_writer.release()
            print("Done processing!")
            break
        frame_rgb = frame[:, :, ::-1]
        frame_img = Image.fromarray(frame_rgb)
        w, h = frame_img.size
        if h > 500:
            frame_img = frame_img.resize((int(w * 0.5), int(h * 0.5)))
        if i % 2 == 0:
            boxes = detector.detect(frame_img)
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        vid_writer.write(frame.astype(np.uint8))
        cv2.imshow("MtCnn detector", frame)
        print(size)
        print("fps:", (i + 1) / (time.time() - start_time))
        print(round(i * 100 / count, 2), "%")
        i += 1
