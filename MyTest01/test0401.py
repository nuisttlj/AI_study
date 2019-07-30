import os
from PIL import Image, ImageDraw
import traceback
import time
import cv2
import numpy as np

anno_src = r"D:\celebA\Anno\list_bbox_celeba.txt"
img_dir = r"D:\celebA\img_celeba"

for i, line in enumerate(open(anno_src)):
    if i < 2:
        continue
    else:
        strs = line.strip().split()
        image_filename = strs[0].strip()
        image_file = os.path.join(img_dir, image_filename)
        # print(image_file)

        img = Image.open(image_file)
        img_w, img_h = img.size
        x1 = float(strs[1].strip())
        y1 = float(strs[2].strip())
        w = float(strs[3].strip())
        h = float(strs[4].strip())
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        new_w, new_h = w * 0.9, h * 0.75
        x1 = center_x - new_w / 2
        y1 = center_y - h / 2
        x2 = center_x + new_w / 2
        y2 = center_y + new_h / 2
        # x2 = x1 + w
        # y2 = y1 + h
        img_draw = ImageDraw.Draw(img)
        img_draw.rectangle((x1, y1, x2, y2), outline=(0, 0, 255))
        img_data = np.array(img)
        img_data_cv2 = img_data[:, :, ::-1]
        if max(img_w, img_h) > 500:
            scale = max(img_w, img_h) / 500
            img_data_cv2 = cv2.resize(img_data_cv2, (int(img_w / scale), int(img_h / scale)))
        cv2.imshow("celebA_img", img_data_cv2)
        cv2.waitKey(1000)
        # cv2.destroyAllWindows()

# a = [1, 2, 3, 4]
# for i in a:
#     a1 = a.copy()
#     a1.remove(i)
#     for j in a1:
#         a2 = a1.copy()
#         a2.remove(j)
#         for k in a2:
#             num = i * 100 + j * 10 + k
#             print(num)

# a = ["1 2 3 4", "2,3,4,6", "3,6,8,0"]
# print(a[0].split())

