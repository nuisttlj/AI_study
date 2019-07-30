import os
from PIL import Image
import numpy as np
import utils
import traceback

anno_src = r"D:\celebA\Anno\list_bbox_celeba.txt"
img_dir = r"D:\celebA\img_celeba"

save_path = r"C:\mycelebA_dev"

# 标签  0 只用于分类  1 用于分类和回归  2 只用于回归

for face_size in [24]:
    print("gen {} image".format(face_size))
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        anno_src_open = open(anno_src, "r")
        anno_src_num = len(anno_src_open.readlines()) - 2
        for i, line in enumerate(open(anno_src)):
            if i < 165000:
                continue
            try:
                strs = line.strip().split()
                image_filename = strs[0].strip()
                image_file = os.path.join(img_dir, image_filename)

                with Image.open(image_file) as img:
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
                    # if max(new_w, new_h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    #     continue
                    if max(w, h) < 20 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]

                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    # for _ in range(2):
                    #     w_ = np.random.randint(np.floor(-w * 0.1), np.ceil(w * 0.1))
                    #     h_ = np.random.randint(np.floor(-h * 0.1), np.ceil(h * 0.1))
                    #     cx_ = cx + w_
                    #     cy_ = cy + h_
                    #
                    #     side_len = np.random.randint(np.floor(min(w, h) * 0.9), np.ceil(1.1 * max(w, h)))
                    #     x1_ = max(cx_ - side_len / 2, 0)
                    #     y1_ = max(cy_ - side_len / 2, 0)
                    #     x2_ = min(x1_ + side_len, img_w)
                    #     y2_ = min(y1_ + side_len, img_h)
                    #
                    #     crop_box = np.array([x1_, y1_, x2_, y2_])
                    #
                    #     offset_x1 = (x1 - x1_) / side_len
                    #     offset_y1 = (y1 - y1_) / side_len
                    #     offset_x2 = (x2 - x2_) / side_len
                    #     offset_y2 = (y2 - y2_) / side_len
                    #
                    #     face_crop = img.crop(crop_box)
                    #     face_resize = face_crop.resize((face_size, face_size))
                    #
                    #     iou = utils.iou(crop_box, np.array(boxes))[0]
                    #     if iou >= 0.7:
                    #         positive_anno_file.write(
                    #             "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(positive_count, 1, offset_x1, offset_y1,
                    #                                                             offset_x2, offset_y2))
                    #         positive_anno_file.flush()
                    #         face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                    #         positive_count += 1
                    #     elif iou > 0.4 and iou < 0.7:
                    #         part_anno_file.write(
                    #             "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                    #                                                         offset_x2, offset_y2))
                    #         part_anno_file.flush()
                    #         face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                    #         part_count += 1
                    #     elif iou < 0.25:
                    #         negative_anno_file.write(
                    #             "negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                    #         negative_anno_file.flush()
                    #         face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    #         negative_count += 1

                    # 生成正样本
                    for _ in range(1):
                        side_len = np.random.randint(min(w, h), max(w, h))
                        x1_ = max(cx - side_len / 2, 0)
                        y1_ = max(cy - side_len / 2, 0)
                        x2_ = min(x1_ + side_len, img_w)
                        y2_ = min(y1_ + side_len, img_h)

                        crop_box = np.array([x1_, y1_, x2_, y2_])

                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size))

                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        if iou > 0.7:
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6}\n".format(positive_count, 1, offset_x1,
                                                                                    offset_y1, offset_x2, offset_y2, 0))
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
                        # elif iou > 0.2 and iou < 0.4:
                        #     part_anno_file.write(
                        #         "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                        #                                                     offset_x2, offset_y2))
                        #     part_anno_file.flush()
                        #     face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                        #     part_count += 1

                    # 生成负样本
                    # 采集耳朵
                    if i % 2 == 0:
                        for _ in range(1):
                            side_len = np.random.randint(8, 24)
                            x1_ = np.random.randint(max(x1 - side_len, 0), min(x1 + side_len, x2))
                            y1_ = np.random.randint(int(y1), max(2 * y2 / 3 - side_len, y1 + 1))
                            x2_ = x1_ + side_len
                            y2_ = y1_ + side_len
                            crop_box = np.array([x1_, y1_, x2_, y2_])
                            # offset_x1 = (x1 - x1_) / side_len
                            # offset_y1 = (y1 - y1_) / side_len
                            # offset_x2 = (x2 - x2_) / side_len
                            # offset_y2 = (y2 - y2_) / side_len
                            face_crop = img.crop(crop_box)
                            iou = utils.iou(crop_box, np.array(boxes))[0]
                            # if iou > 0.4 and iou < 0.7:
                            #     face_resize = face_crop.resize((face_size, face_size))
                            #     part_anno_file.write(
                            #         "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                            #                                                     offset_x2, offset_y2))
                            #     part_anno_file.flush()
                            #     face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            #     part_count += 1
                            if iou < 0.25:
                                face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                                negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 {2}\n".format(negative_count, 0, 0))
                                negative_anno_file.flush()
                                face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                                negative_count += 1
                    else:
                        # 采集耳朵
                        for _ in range(1):
                            side_len = np.random.randint(8, 24)
                            x1_ = np.random.randint(max(x2 - side_len, x1), min(x2 + side_len, img_w))
                            y1_ = np.random.randint(int(y1), max(2 * y2 / 3 - side_len, y1 + 1))
                            x2_ = x1_ + side_len
                            y2_ = y1_ + side_len
                            crop_box = np.array([x1_, y1_, x2_, y2_])
                            # offset_x1 = (x1 - x1_) / side_len
                            # offset_y1 = (y1 - y1_) / side_len
                            # offset_x2 = (x2 - x2_) / side_len
                            # offset_y2 = (y2 - y2_) / side_len
                            face_crop = img.crop(crop_box)
                            iou = utils.iou(crop_box, np.array(boxes))[0]
                            # if iou > 0.4 and iou < 0.7:
                            #     face_resize = face_crop.resize((face_size, face_size))
                            #     part_anno_file.write(
                            #         "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                            #                                                     offset_x2, offset_y2))
                            #     part_anno_file.flush()
                            #     face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            #     part_count += 1
                            if iou < 0.25:
                                face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                                negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 {2}\n".format(negative_count, 0, 0))
                                negative_anno_file.flush()
                                face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                                negative_count += 1

                    # for _ in range(1):
                    #     side_len = np.random.randint(20, 60)
                    #     x2_ = np.random.randint(int(min(x2, img_w - 1)), img_w)
                    #     y2_ = np.random.randint(0 + side_len, img_h)
                    #     x1_ = x2_ - side_len
                    #     y1_ = y2_ - side_len
                    #     crop_box = np.array([x1_, y1_, x2_, y2_])
                    #     # offset_x1 = (x1 - x1_) / side_len
                    #     # offset_y1 = (y1 - y1_) / side_len
                    #     # offset_x2 = (x2 - x2_) / side_len
                    #     # offset_y2 = (y2 - y2_) / side_len
                    #     face_crop = img.crop(crop_box)
                    #     iou = utils.iou(crop_box, np.array(boxes))[0]
                    #     # if iou > 0.4 and iou < 0.7:
                    #     #     face_resize = face_crop.resize((face_size, face_size))
                    #     #     part_anno_file.write(
                    #     #         "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                    #     #                                                     offset_x2, offset_y2))
                    #     #     part_anno_file.flush()
                    #     #     face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                    #     #     part_count += 1
                    #     if iou < 0.25:
                    #         face_crop = img.crop(crop_box)
                    #         face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                    #         negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 {2}\n".format(negative_count, 0, 0))
                    #         negative_anno_file.flush()
                    #         face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    #         negative_count += 1

                    # for _ in range(1):
                    #     side_len = np.random.randint(face_size, int(max(min(img_w / 2, img_h / 2), face_size + 1)))
                    #     y1_ = np.random.randint(0, int(max(y1, 1)))
                    #     x1_ = np.random.randint(0, img_w - side_len)
                    #     y2_ = y1_ + side_len
                    #     x2_ = x1_ + side_len
                    #     crop_box = np.array([x1_, y1_, x2_, y2_])
                    #     # offset_x1 = (x1 - x1_) / side_len
                    #     # offset_y1 = (y1 - y1_) / side_len
                    #     # offset_x2 = (x2 - x2_) / side_len
                    #     # offset_y2 = (y2 - y2_) / side_len
                    #     face_crop = img.crop(crop_box)
                    #     iou = utils.iou(crop_box, np.array(boxes))[0]
                    #     # if iou > 0.4 and iou < 0.7:
                    #     #     face_resize = face_crop.resize((face_size, face_size))
                    #     #     part_anno_file.write(
                    #     #         "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                    #     #                                                     offset_x2, offset_y2))
                    #     #     part_anno_file.flush()
                    #     #     face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                    #     #     part_count += 1
                    #     if iou < 0.25:
                    #         face_crop = img.crop(crop_box)
                    #         face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                    #         negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 {2}\n".format(negative_count, 0, 0))
                    #         negative_anno_file.flush()
                    #         face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    #         negative_count += 1

                    # for _ in range(1):
                    #     side_len = np.random.randint(20, 60)
                    #     y2_ = np.random.randint(int(min(y2, img_h - 1)), img_h)
                    #     x2_ = np.random.randint(0 + side_len, img_w)
                    #     x1_ = x2_ - side_len
                    #     y1_ = y2_ - side_len
                    #     crop_box = np.array([x1_, y1_, x2_, y2_])
                    #     # offset_x1 = (x1 - x1_) / side_len
                    #     # offset_y1 = (y1 - y1_) / side_len
                    #     # offset_x2 = (x2 - x2_) / side_len
                    #     # offset_y2 = (y2 - y2_) / side_len
                    #     face_crop = img.crop(crop_box)
                    #     iou = utils.iou(crop_box, np.array(boxes))[0]
                    #     # if iou > 0.35 and iou < 0.7:
                    #     #     face_resize = face_crop.resize((face_size, face_size))
                    #     #     part_anno_file.write(
                    #     #         "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                    #     #                                                     offset_x2, offset_y2))
                    #     #     part_anno_file.flush()
                    #     #     face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                    #     #     part_count += 1
                    #     if iou <= 0.25:
                    #         face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                    #         negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 {2}\n".format(negative_count, 0, 0))
                    #         negative_anno_file.flush()
                    #         face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    #         negative_count += 1

            except Exception as e:
                traceback.print_exc()

            print("进度: {0}%".format(round((i + 1) * 100 / anno_src_num), 2))

        # nonface_source_path = r"D:/image/"
        # picname_list = os.listdir(nonface_source_path)
        # for picname in picname_list:
        #     imgdir = os.path.join(nonface_source_path, picname)
        #     with Image.open(imgdir) as img:
        #         img_w, img_h = img.size
        #         for _ in range(3):
        #             side_len = np.random.randint(face_size, max(min(img_w, img_h) / 2, face_size + 1))
        #             x1 = np.random.randint(0, img_w - side_len)
        #             y1 = np.random.randint(0, img_h - side_len)
        #             crop_box = np.array([x1, y1, x1 + side_len, y1 + side_len])
        #             face_crop = img.crop(crop_box)
        #             face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
        #             try:
        #                 negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
        #                 negative_anno_file.flush()
        #                 face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
        #             except:
        #                 continue
        #             negative_count += 1

        # nonface_source_path = r"D:/hand_image/"
        # picname_list = os.listdir(nonface_source_path)
        # for picname in picname_list:
        #     imgdir = os.path.join(nonface_source_path, picname)
        #     with Image.open(imgdir) as img:
        #         img_w, img_h = img.size
        #         for _ in range(6):
        #             side_len = np.random.randint(face_size, max(min(img_w, img_h) / 2, face_size + 1))
        #             x1 = np.random.randint(0, img_w - side_len)
        #             y1 = np.random.randint(0, img_h - side_len)
        #             crop_box = np.array([x1, y1, x1 + side_len, y1 + side_len])
        #             face_crop = img.crop(crop_box)
        #             face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
        #             try:
        #                 negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
        #                 negative_anno_file.flush()
        #                 face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
        #             except:
        #                 continue
        #             negative_count += 1

    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()
