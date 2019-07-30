import os
import shutil


for face_size in [12, 24, 48]:
    for lable_type in ["negative"]:
        source_dir = r"C:\mywiderface_train_bg\{0}\{1}".format(face_size, lable_type)
        source_file = r"C:\mywiderface_train_bg\{0}\{1}.txt".format(face_size, lable_type)
        dest_dir = r"C:\mywiderface_train\{0}\{1}".format(face_size, lable_type)
        dest_file = r"C:\mywiderface_train\{0}\{1}.txt".format(face_size, lable_type)
        label_file = open(source_file, "r")
        lable_list = label_file.readlines()
        # pic_list = [string.strip().split()[0].replace("{0}/".format(lable_type), "") for string in
        #                      lable_list]
        # if not os.path.exists(dest_dir):
        #     os.makedirs(dest_dir)
        target_label_file = open(dest_file, "a")
        for line in lable_list:
            new_line = line.replace("/", "/ear")
            target_label_file.write(new_line)
        target_label_file.close()
        target_sum = len(open(dest_file, "r").readlines())
        print("total_num = ", target_sum)

        pic_name = os.listdir(source_dir)
        for pic in pic_name:
            new_picname = "ear"+pic
            shutil.move(os.path.join(source_dir, pic), os.path.join(dest_dir, new_picname))

        pic_num = len(os.listdir(dest_dir))
        print(pic_num)
        print("==============================")