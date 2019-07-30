# 标签  0 只用于分类  1 用于分类和回归  2 只用于回归
for face_size in [12, 24, 48]:
    for lable_type in ["positive", "part", "negative"]:
        if lable_type == "positive":
            used_label = " 1"

        elif lable_type == "part":
            used_label = " 2"

        elif lable_type == "negative":
            used_label = " 0"

        source_dir = r"C:\mywiderface_train_bg\{0}\{1}".format(face_size, lable_type)
        source_file = r"C:\mywiderface_train_bg\{0}\{1}.txt".format(face_size, lable_type)
        label_file = open(source_file, "r")
        lable_list = label_file.readlines()
        print(len(lable_list))
        target_label_file = open(source_file, "w")
        for line in lable_list:
            new_line = line.strip() + used_label + "\n"
            target_label_file.write(new_line)
        target_label_file.close()
        target_sum = len(open(source_file, "r").readlines())
        print("total_num = ", target_sum)
