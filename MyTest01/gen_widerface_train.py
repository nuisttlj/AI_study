import os
from PIL import Image
import numpy as np
import utils
import traceback

anno_src = r"C:\widerface\wider_face_split\wider_face_train_bbx_gt.txt"
img_dir = r"C:\widerface\WIDER_train\images"

train_save_path = r"C:\mywiderface_train"

for face_size in [12, 24, 48]:
    print("gen {} image".format(face_size))
    positive_image_dir = os.path.join(train_save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(train_save_path, str(face_size), "negative")
    part_image_dir = os.path.join(train_save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    positive_anno_filename = os.path.join(train_save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(train_save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(train_save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        all_strs = open(anno_src).readlines()
        for i in range(len(all_strs)):
            try:
                if ".jpg" in all_strs[i]:
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

                        # print(i, "--", bbox_num, "--", len(boxes))
                        if not boxes:
                            continue
                        for box in boxes:
                            boxes_copy = boxes.copy()
                            boxes_copy.remove(box)
                            x1 = box[0]
                            y1 = box[1]
                            x2 = box[2]
                            y2 = box[3]
                            w = x2 - x1
                            h = y2 - y1
                            cx = x1 + w / 2
                            cy = y1 + h / 2

                            # 生成正样本和部分样本以及少量负样本 主要生成正样本
                            for _ in range(1):
                                if max(w, h) < 20:
                                    continue

                                w_ = np.random.randint(np.floor(-w * 0.05), np.ceil(w * 0.05))
                                h_ = np.random.randint(np.floor(-h * 0.05), np.ceil(h * 0.05))
                                cx_ = cx + w_
                                cy_ = cy + h_

                                side_len = np.random.randint(np.ceil((w + h) / 2), np.ceil(1.1 * max(w, h)))

                                if cx_ > img_w or cy_ > img_h:
                                    continue

                                x1_ = int(max(cx_ - side_len / 2, 0))
                                y1_ = int(max(cy_ - side_len / 2, 0))
                                x2_ = min(x1_ + side_len, img_w)
                                y2_ = min(y1_ + side_len, img_h)

                                crop_box = np.array([x1_, y1_, x2_, y2_])

                                if boxes_copy:
                                    if np.max(utils.iou(crop_box, np.array(boxes_copy))) > 0:
                                        continue

                                offset_x1 = (x1 - x1_) / side_len
                                offset_y1 = (y1 - y1_) / side_len
                                offset_x2 = (x2 - x2_) / side_len
                                offset_y2 = (y2 - y2_) / side_len

                                face_crop = img.crop(crop_box)
                                face_resize = face_crop.resize((face_size, face_size))

                                iou = utils.iou(crop_box, np.array([box]))[0]
                                if iou > 0.65:
                                    positive_anno_file.write(
                                        "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(positive_count, 1, offset_x1,
                                                                                        offset_y1,
                                                                                        offset_x2, offset_y2))
                                    positive_anno_file.flush()
                                    face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                                    positive_count += 1
                                elif 0.4 < iou <= 0.65:
                                    part_anno_file.write(
                                        "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                                                                                    offset_x2, offset_y2))
                                    part_anno_file.flush()
                                    face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                                    part_count += 1
                                elif iou < 0.3:
                                    negative_anno_file.write(
                                        "negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                                    negative_anno_file.flush()
                                    face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                                    negative_count += 1

                            # if positive_count > 5000:
                            #     print(positive_count, part_count, negative_count)

                            # 生成正样本和部分样本以及少量负样本 主要生成部分样本
                            for _ in range(1):
                                if max(w, h) < 20:
                                    continue

                                w_ = np.random.randint(np.floor(-w * 0.2), np.ceil(w * 0.2))
                                h_ = np.random.randint(np.floor(-h * 0.25), np.ceil(h * 0.25))
                                cx_ = cx + w_
                                cy_ = cy + h_

                                side_len = np.random.randint(np.floor(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                                if cx_ > img_w or cy_ > img_h:
                                    continue

                                x1_ = int(max(cx_ - side_len / 2, 0))
                                y1_ = int(max(cy_ - side_len / 2, 0))
                                x2_ = min(x1_ + side_len, img_w)
                                y2_ = min(y1_ + side_len, img_h)

                                crop_box = np.array([x1_, y1_, x2_, y2_])

                                if boxes_copy:
                                    if np.max(utils.iou(crop_box, np.array(boxes_copy))) > 0:
                                        continue

                                offset_x1 = (x1 - x1_) / side_len
                                offset_y1 = (y1 - y1_) / side_len
                                offset_x2 = (x2 - x2_) / side_len
                                offset_y2 = (y2 - y2_) / side_len

                                face_crop = img.crop(crop_box)
                                face_resize = face_crop.resize((face_size, face_size))

                                iou = utils.iou(crop_box, np.array([box]))[0]
                                if iou > 0.65:
                                    positive_anno_file.write(
                                        "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(positive_count, 1, offset_x1,
                                                                                        offset_y1,
                                                                                        offset_x2, offset_y2))
                                    positive_anno_file.flush()
                                    face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                                    positive_count += 1
                                elif 0.4 < iou <= 0.65:
                                    part_anno_file.write(
                                        "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                                                                                    offset_x2, offset_y2))
                                    part_anno_file.flush()
                                    face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                                    part_count += 1
                                elif iou < 0.3:
                                    negative_anno_file.write(
                                        "negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                                    negative_anno_file.flush()
                                    face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                                    negative_count += 1

                            # if positive_count > 6000:
                            #     print(positive_count, part_count, negative_count)
                            #     exit()

                            # 生成负样本
                            for _ in range(2):
                                w_ = np.random.randint(np.floor(-w * 0.5), np.ceil(w * 0.5))
                                h_ = np.random.randint(np.floor(-h * 0.5), np.ceil(h * 0.5))
                                cx_ = cx + w_
                                cy_ = cy + h_

                                side_len = np.random.randint(face_size,
                                                             max(int(max(min(img_w / 4, img_h / 4), min(w, h))),
                                                                 face_size + 1))

                                if cx_ > img_w or cy_ > img_h:
                                    continue

                                x1_ = int(max(cx_ - side_len / 2, 0))
                                y1_ = int(max(cy_ - side_len / 2, 0))
                                x2_ = min(x1_ + side_len, img_w)
                                y2_ = min(y1_ + side_len, img_h)

                                crop_box = np.array([x1_, y1_, x2_, y2_])

                                # if boxes_copy:
                                #     if np.max(utils.iou(crop_box, np.array(boxes_copy))) > 0:
                                #         continue

                                face_crop = img.crop(crop_box)
                                iou = np.max(utils.iou(crop_box, np.array(boxes)))
                                if iou < 0.3:
                                    face_resize = face_crop.resize((face_size, face_size))
                                    negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                                    negative_anno_file.flush()
                                    face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                                    negative_count += 1

                            # if positive_count > 6000:
                            #     print(positive_count, part_count, negative_count)
                            #     exit()

                            # for _ in range(1):
                            #     side_len = np.random.randint(face_size, int(max(min(w / 2, h / 2), face_size + 1)))
                            #     center_x_ = np.random.randint(
                            #         min(int(x2 - side_len / 2), img_w - int(side_len / 2) - 1),
                            #         min(int(x2 + side_len / 2), img_w - int(side_len / 2)))
                            #     center_y_ = np.random.randint(max(int(y1 - side_len / 2), int(side_len / 2)),
                            #                                   min(int(y2 + side_len / 2), img_h - int(side_len / 2)))
                            #     x1_ = center_x_ - int(side_len / 2)
                            #     y1_ = center_y_ - int(side_len / 2)
                            #     x2_ = x1_ + side_len
                            #     y2_ = y1_ + side_len
                            #
                            #     crop_box = np.array([x1_, y1_, x2_, y2_])
                            #
                            #     if boxes_copy:
                            #         if np.max(utils.iou(crop_box, np.array(boxes_copy))) > 0:
                            #             continue
                            #
                            #     offset_x1 = (x1 - x1_) / side_len
                            #     offset_y1 = (y1 - y1_) / side_len
                            #     offset_x2 = (x2 - x2_) / side_len
                            #     offset_y2 = (y2 - y2_) / side_len
                            #     face_crop = img.crop(crop_box)
                            #     iou = utils.iou(crop_box, np.array([box]))[0]
                            #     if 0.4 < iou <= 0.65:
                            #         face_resize = face_crop.resize((face_size, face_size))
                            #         part_anno_file.write(
                            #             "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                            #                                                         offset_x2, offset_y2))
                            #         part_anno_file.flush()
                            #         face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            #         part_count += 1
                            #     elif iou < 0.3:
                            #         face_crop = img.crop(crop_box)
                            #         face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                            #         negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            #         negative_anno_file.flush()
                            #         face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            #         negative_count += 1
                            #
                            # for _ in range(1):
                            #     side_len = np.random.randint(face_size, int(max(min(w / 2, h / 2), face_size + 1)))
                            #     center_x_ = np.random.randint(max(int(x1 - side_len / 2), int(side_len / 2)),
                            #                                   min(int(x2 + side_len / 2), img_w - int(side_len / 2)))
                            #     center_y_ = np.random.randint(max(int(y1 - side_len / 2), int(side_len / 2)),
                            #                                   int(y1 + side_len / 2))
                            #     x1_ = center_x_ - int(side_len / 2)
                            #     y1_ = center_y_ - int(side_len / 2)
                            #     x2_ = x1_ + side_len
                            #     y2_ = y1_ + side_len
                            #
                            #     crop_box = np.array([x1_, y1_, x2_, y2_])
                            #
                            #     if boxes_copy:
                            #         if np.max(utils.iou(crop_box, np.array(boxes_copy))) > 0:
                            #             continue
                            #
                            #     offset_x1 = (x1 - x1_) / side_len
                            #     offset_y1 = (y1 - y1_) / side_len
                            #     offset_x2 = (x2 - x2_) / side_len
                            #     offset_y2 = (y2 - y2_) / side_len
                            #     face_crop = img.crop(crop_box)
                            #     iou = utils.iou(crop_box, np.array([box]))[0]
                            #     if 0.4 < iou <= 0.65:
                            #         face_resize = face_crop.resize((face_size, face_size))
                            #         part_anno_file.write(
                            #             "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                            #                                                         offset_x2, offset_y2))
                            #         part_anno_file.flush()
                            #         face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            #         part_count += 1
                            #     elif iou < 0.3:
                            #         face_crop = img.crop(crop_box)
                            #         face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                            #         negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            #         negative_anno_file.flush()
                            #         face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            #         negative_count += 1
                            #
                            # for _ in range(1):
                            #     side_len = np.random.randint(face_size, int(max(min(w / 2, h / 2), face_size + 1)))
                            #     center_x_ = np.random.randint(max(int(x1 - side_len / 2), int(side_len / 2)),
                            #                                   max(int(x2 + side_len / 2), img_w - int(side_len / 2)))
                            #     center_y_ = np.random.randint(
                            #         min(int(y2 - side_len / 2), img_h - int(side_len / 2) - 1),
                            #         max(int(y2 + side_len / 2), img_h - int(side_len / 2)))
                            #     x1_ = center_x_ - int(side_len / 2)
                            #     y1_ = center_y_ - int(side_len / 2)
                            #     x2_ = x1_ + side_len
                            #     y2_ = y1_ + side_len
                            #
                            #     crop_box = np.array([x1_, y1_, x2_, y2_])
                            #
                            #     if boxes_copy:
                            #         if np.max(utils.iou(crop_box, np.array(boxes_copy))) > 0:
                            #             continue
                            #
                            #     offset_x1 = (x1 - x1_) / side_len
                            #     offset_y1 = (y1 - y1_) / side_len
                            #     offset_x2 = (x2 - x2_) / side_len
                            #     offset_y2 = (y2 - y2_) / side_len
                            #     face_crop = img.crop(crop_box)
                            #     iou = utils.iou(crop_box, np.array([box]))[0]
                            #     if 0.4 < iou <= 0.65:
                            #         face_resize = face_crop.resize((face_size, face_size))
                            #         part_anno_file.write(
                            #             "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                            #                                                         offset_x2, offset_y2))
                            #         part_anno_file.flush()
                            #         face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            #         part_count += 1
                            #     elif iou < 0.3:
                            #         face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                            #         negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            #         negative_anno_file.flush()
                            #         face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            #         negative_count += 1

            except Exception as e:
                traceback.print_exc()

        # 单独生成手部负样本
        nonface_source_path = r"D:/hand_image_train/"
        picname_list = os.listdir(nonface_source_path)
        for picname in picname_list:
            imgdir = os.path.join(nonface_source_path, picname)
            with Image.open(imgdir) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size
                for _ in range(6):
                    side_len = np.random.randint(face_size, max(min(img_w, img_h) / 2, face_size + 1))
                    x1 = np.random.randint(0, img_w - side_len)
                    y1 = np.random.randint(0, img_h - side_len)
                    crop_box = np.array([x1, y1, x1 + side_len, y1 + side_len])
                    face_crop = img.crop(crop_box)
                    face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                    try:
                        negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                        negative_anno_file.flush()
                        face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    except:
                        continue
                    negative_count += 1

    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()
