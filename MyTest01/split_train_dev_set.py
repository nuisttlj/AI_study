import os
import shutil
import random

# 范围取
for face_size in [48]:
    for lable_type in ["positive"]:
        source_dir = r"C:\mywiderface_train\{0}\{1}".format(face_size, lable_type)
        source_file = r"C:\mywiderface_train\{0}\{1}.txt".format(face_size, lable_type)
        dest_dir = r"C:\mywiderface_dev\{0}\{1}".format(face_size, lable_type)
        dest_file = r"C:\mywiderface_dev\{0}\{1}.txt".format(face_size, lable_type)
        label_file = open(source_file, "r")
        lable_list = label_file.readlines()
        total_num = len(lable_list)
        pre_dev_sum = len(open(dest_file, "r").readlines())
        # selected_lable = lable_list[723886:743886]
        selected_lable = lable_list[total_num-10000:total_num]
        selected_pic_list = [string.strip().split()[0].replace("{0}/".format(lable_type), "") for string in
                             selected_lable]
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        train_label_file = open(source_file, "w")
        dev_label_file = open(dest_file, "a")
        for line in lable_list:
            if line in selected_lable:
                dev_label_file.write(line)
            else:
                train_label_file.write(line)
        train_label_file.close()
        dev_label_file.close()
        train_sum = len(open(source_file, "r").readlines())
        dev_sum = len(open(dest_file, "r").readlines())
        print(total_num, "=", train_sum + dev_sum - pre_dev_sum, "=", train_sum, "+", dev_sum, "-", pre_dev_sum)

        pic_name = os.listdir(source_dir)
        for pic in pic_name:
            if pic in selected_pic_list:
                shutil.move(os.path.join(source_dir, pic), dest_dir)

# 打乱取
# for face_size in [12, 24, 48]:
#     for lable_type in ["positive", "part", "negative"]:
#         source_dir = r"E:\myceleba\{0}\{1}".format(face_size, lable_type)
#         source_file = r"E:\myceleba\{0}\{1}.txt".format(face_size, lable_type)
#         dest_dir = r"E:\myceleba_dev\{0}\{1}".format(face_size, lable_type)
#         dest_file = r"E:\myceleba_dev\{0}\{1}.txt".format(face_size, lable_type)
#         label_file = open(source_file, "r")
#         lable_list = label_file.readlines()
#         total_num = len(lable_list)
#         random_lable = random.sample(lable_list, 2500)
#         ran_pic_list = [string.strip().split()[0].replace("{0}/".format(lable_type), "") for string in random_lable]
#         if not os.path.exists(dest_dir):
#             os.makedirs(dest_dir)
#         train_label_file = open(source_file, "w")
#         dev_label_file = open(dest_file, "a")
#         for line in lable_list:
#             if line in random_lable:
#                 dev_label_file.write(line)
#             else:
#                 train_label_file.write(line)
#         train_label_file.close()
#         dev_label_file.close()
#         train_sum = len(open(source_file, "r").readlines())
#         dev_sum = len(open(dest_file, "r").readlines())
#         print(total_num, "=", train_sum + dev_sum - 2500, "=", train_sum, "+", dev_sum)
#
#         pic_name = os.listdir(source_dir)
#         for pic in pic_name:
#             if pic in ran_pic_list:
#                 shutil.move(os.path.join(source_dir, pic), dest_dir)
