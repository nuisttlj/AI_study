import os
from PIL import Image, ImageDraw
import traceback
import time
import cv2
import numpy as np

anno_src = r"D:\widerface\wider_face_split\wider_face_train_bbx_gt.txt"
img_dir = r"D:\widerface\WIDER_train\images"

all_strs = open(anno_src).readlines()
# print(all_strs)
# exit()
for i in range(len(all_strs)):

    # strs = line.strip().split()
    # print(strs)
    # # while i == 10:
    # #     exit()
    if ".jpg" in all_strs[i]:
        image_filename = all_strs[i].strip()
        image_file = os.path.join(img_dir, image_filename)
        img = cv2.imread(image_file)
        bbox_num = int(all_strs[i+1])
        # if bbox_num < 10:
        for j in range(bbox_num):
            pic_str = all_strs[i+1+j+1]
            pic_str_list = pic_str.strip().split()
            # img = Image.open(image_file)
            img_w, img_h = img.shape[0], img.shape[1]
            x1 = int(pic_str_list[0].strip())
            y1 = int(pic_str_list[1].strip())
            w = int(pic_str_list[2].strip())
            h = int(pic_str_list[3].strip())
            x2 = x1 + w
            y2 = y1 + h
            # img_draw = ImageDraw.Draw(img)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0))
            # img_data = np.array(img)
            # img_data_cv2 = img_data[:, :, ::-1]
            # if max(img_w, img_h) > 500:
            #     scale = max(img_w, img_h) / 500
            #     img_data_cv2 = cv2.resize(img_data_cv2, (int(img_w / scale), int(img_h / scale)))
        cv2.namedWindow("widerface_img")
        cv2.imshow("widerface_img", img)
        cv2.waitKey(0)