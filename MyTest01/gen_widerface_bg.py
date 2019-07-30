import os
from PIL import Image
import numpy as np
import utils
import traceback

anno_src = r"C:\widerface\wider_face_split\wider_face_train_bbx_gt.txt"
img_dir = r"C:\widerface\WIDER_train\images"

train_save_path = r"C:\mywiderface_train_bg"

for face_size in [12, 24, 48]:
    print("gen {} image".format(face_size))
    negative_image_dir = os.path.join(train_save_path, str(face_size), "negative")

    for dir_path in [negative_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    negative_anno_filename = os.path.join(train_save_path, str(face_size), "negative.txt")

    negative_count = 0

    try:
        negative_anno_file = open(negative_anno_filename, "w")

        # all_strs = open(anno_src).readlines()
        # str_length = len(all_strs)
        # for i in range(str_length):
        #     try:
        #         if ".jpg" in all_strs[i]:
        #             image_filename = all_strs[i].strip()
        #             image_file = os.path.join(img_dir, image_filename)
        #             with Image.open(image_file) as img:
        #                 boxes = []
        #                 img_w, img_h = img.size
        #                 bbox_num = int(all_strs[i + 1])
        #                 # if bbox_num < 10:
        #                 for j in range(bbox_num):
        #                     pic_str = all_strs[i + 1 + j + 1]
        #                     pic_str_list = pic_str.strip().split()
        #                     x1 = int(pic_str_list[0].strip())
        #                     y1 = int(pic_str_list[1].strip())
        #                     w = int(pic_str_list[2].strip())
        #                     h = int(pic_str_list[3].strip())
        #                     x2 = x1 + w
        #                     y2 = y1 + h
        #
        #                     if x1 <= 0 or y1 <= 0 or w <= 0 or h <= 0:
        #                         continue
        #
        #                     boxes.append([x1, y1, x2, y2])
        #
        #                 # print(i, "--", bbox_num, "--", len(boxes))
        #                 if not boxes:
        #                     continue
        #                 for box in boxes:
        #                     x1 = box[0]
        #                     y1 = box[1]
        #                     x2 = box[2]
        #                     y2 = box[3]
        #                     w = x2 - x1
        #                     h = y2 - y1
        #                     cx = x1 + w / 2
        #                     cy = y1 + h / 2

                        # 生成负样本
            #             for _ in range(8):
            #                 if face_size == 48:
            #                     # side_len = np.random.randint(40, face_size)
            #                     side_len = np.random.randint(10, 40)
            #                 elif face_size == 24:
            #                     side_len = np.random.randint(8, 24)
            #                 else:
            #                     side_len = face_size
            #
            #                 x1_ = np.random.randint(0, img_w - side_len)
            #                 y1_ = np.random.randint(0, img_h - side_len)
            #                 x2_ = x1_ + side_len
            #                 y2_ = y1_ + side_len
            #
            #                 crop_box = np.array([x1_, y1_, x2_, y2_])
            #
            #                 if np.max(utils.iou(crop_box, np.array(boxes))) > 0:
            #                     continue
            #
            #                 face_crop = img.crop(crop_box)
            #                 face_resize = face_crop.resize((face_size, face_size))
            #                 negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 0\n".format(negative_count, 0))
            #                 negative_anno_file.flush()
            #                 face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
            #                 negative_count += 1
            #
            # except Exception as e:
            #     traceback.print_exc()
            #
            # print("gen{1} widerface进度 {0}%".format(round((i+1) * 100 / str_length, 2), face_size))

        # 单独生成爬取的背景负样本
        nonface_source_path = r"C:\earimage"
        picname_list = os.listdir(nonface_source_path)
        for picname in picname_list:
            imgdir = os.path.join(nonface_source_path, picname)
            with Image.open(imgdir) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size
                for _ in range(100):
                    # side_len = np.random.randint(face_size, max(min(img_w, img_h) / 2, face_size + 1))
                    # side_len = np.random.randint(8, 48)
                    side_len = np.random.randint(min(img_w * 0.8, img_h * 0.8), int((img_h + img_w) / 2))
                    x1 = np.random.randint(0, max(img_w - side_len, 1))
                    y1 = np.random.randint(0, max(img_h - side_len, 1))
                    crop_box = np.array([x1, y1, x1 + side_len, y1 + side_len])
                    face_crop = img.crop(crop_box)
                    face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                    try:
                        negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 0\n".format(negative_count, 0))
                        negative_anno_file.flush()
                        face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                        negative_count += 1
                    except:
                        continue

    finally:
        negative_anno_file.close()
