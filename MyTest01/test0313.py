import cv2
import PIL.Image as image
# from tensorflow.python.client import device_lib
import numpy as np
import torch
import os
import re
import operator
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from multiprocessing import Manager
from threading import Thread,Lock

# import tensorflow as tf

# print(tf.__version__)

# print(device_lib.list_local_devices())

# pic = cv2.imread(r"D:\PycharmProjects\MyTest01\compounds_pic\1.145.158.241.254.1.png")
# cv2.rectangle(pic,(145,158),(241,254),(255,0,0),2)
# cv2.imshow("pic", pic)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# pic = image.new("RGB",(256,256),color=(255,255,255))
# pic.show()
# print(np.min(100,10))
# print(max(100, 10))
# a = torch.Tensor(np.arange(24).reshape(1,2,3,4))
# print(a)
# print(a.size(0))

# food = open(r"D:\fridge\Haier.txt","w+").read().splitlines()
# print(food)
# print(food[1])
# food.remove("香蕉")
# print(food)
#
# a = torch.Tensor([[1, 2, 3],[2,3,4],[1,5,6]])
# b = torch.lt(a, 4)
# print(b)
# print(torch.nonzero(b))
# print(a[b])
# print(torch.masked_select(a, b))

# a = open(r"E:\myceleba2\12\positive.txt","r")
# print(int(a.readlines()[0].strip().strip().split()[1]))

# p = re.compile(r"a.*b")
# print(p.findall("abbbaabbaabbbbbab"))

# a = [[2, 1], [4, 0], [3, 1]]
# b = sorted(a, key=lambda x: x[0])
# c = sorted(a, key=operator.itemgetter(0))
# print(b)
# print(c)

# cap = cv2.VideoCapture(r"D:\video\1.mp4")
# fps = cap.get(cv2.CAP_PROP_FPS)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# vid_writer = cv2.VideoWriter(r"D:\video\1_out.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
# print(fps, size)
# # print(cap.isOpened())
# i = 0
# while cv2.waitKey(1) < 0:
#     print(round(i * 100 / count, 2), "%")
#     i += 1
#     hasFrame, frame = cap.read()
#     print(hasFrame, frame.shape)
#     exit(0)
#     # cv2.imshow("frame{0}".format(i), frame)
#     # cv2.waitKey(1000)
#     # cv2.destroyAllWindows()
#     if not hasFrame:
#         cv2.waitKey(1000)
#         cap.release()
#         vid_writer.release()
#         print("Done!")
#         break
#     cv2.rectangle(frame, (100, 100), (200, 200), (0, 0, 255), 2)
#     vid_writer.write(frame)

# name_list = os.listdir(r"D:/image/")
# print(os.path.join(r"D:/image/",name_list[0]))

# list_a = range(1, 1000)
# print(random.sample(list_a, 100))

# list_b = ["afa", "fdgsgg", "fsdfs", "hthnj", "htjyj"]
# print(random.sample(list_b, 5))
# print("fdg" in list_b)

# pic_name = os.listdir(r"E:\myceleba\48\positive")
# for pic in pic_name:
#     print(pic.replace(".jpg"))

# part = "part"
# label_file = open(r"E:\myceleba\24\part.txt", "r")
# c_list = label_file.readlines()[10000:10002]
# print(c_list)
# # for i in c_list:
# #     print(i)
# #     print(c_list)
# #     print(i in c_list)
# print([string.strip().split()[0].replace("{0}/".format(part), "") for string in c_list])

# label_file = open(r"E:\myceleba\48\negative - 副本.txt", "r")
# lines = label_file.readlines()
# pic_list = [string.strip().split()[0].replace("negative/", "") for string in lines]
# name_list = os.listdir(r"E:\myceleba\48\negative")
# print(len(name_list))
# print(len(pic_list))
# pic_set = set(pic_list)
# print(len(pic_set))
# for i in pic_list:
#     if i not in name_list:
#         print(i)


# 解决标签重复问题
# for face_size in [12, 24, 48]:
#     for lable_type in ["positive", "part", "negative"]:
#         source_dir = r"E:\myceleba_dev\{0}\{1}".format(face_size, lable_type)
#         source_file = r"E:\myceleba_dev\{0}\{1}.txt".format(face_size, lable_type)
#         label_file = open(source_file, "r")
#         lable_list = label_file.readlines()
#         total_lable = len(lable_list)
#         lable_set = set(lable_list)
#         lable_list = list(lable_set)
#         train_label_file = open(source_file, "w")
#         for line in lable_list:
#             train_label_file.write(line)
#         train_label_file.close()
#         train_sum = len(open(source_file, "r").readlines())
#         pic_name = os.listdir(source_dir)
#         total_pic = len(pic_name)
#         print("pre_lable", total_lable, "total_lable:", train_sum, "total_pic:", total_pic)


# print(np.maximum(np.array([3, 4, 5]), np.array([1, 3, 6])))
# print(np.mean(np.equal(np.array([3, 4, 5]), np.array([1, 4, 6]))))
# print(np.where(np.array([3, 4, 5]) > 3, 1, 0))

# for face_size in [12, 24, 48]:
#     for lable_type in ["negative"]:
#         source_dir = r"E:\myceleba\{0}\{1}".format(face_size, lable_type)
#         dest_dir = r"E:\pic_error\{0}\{1}".format(face_size, lable_type)
#
#         if not os.path.exists(dest_dir):
#             os.makedirs(dest_dir)
#
#         pic_name = os.listdir(source_dir)
#         length = len(pic_name)
#         for pic in pic_name[length - 100000:length]:
#         # for pic in pic_name:
#             pic_data = np.array(Image.open(os.path.join(source_dir, pic)))
#             if pic_data.ndim != 3 or pic_data.shape[2] != 3:
#                 shutil.move(os.path.join(source_dir, pic), dest_dir)
#                 print("need find")
#                 for i in range(1, 1000000):
#                     replace_name = os.path.join(source_dir, str(int(pic.replace(".jpg", "")) - i) + ".jpg")
#                     if os.path.exists(replace_name):
#                         shutil.copy(replace_name, os.path.join(source_dir, pic))
#                         print("successful found")
#                         break

# judge total
# for face_size in [12, 24, 48]:
#     for lable_type in ["positive", "part", "negative"]:
#         source_dir = r"E:\myceleba\{0}\{1}".format(face_size, lable_type)
#         source_file = r"E:\myceleba\{0}\{1}.txt".format(face_size, lable_type)
#         label_file = open(source_file, "r")
#         lable_list = label_file.readlines()
#         total_lable = len(lable_list)
#         pic_name = os.listdir(source_dir)
#         total_pic = len(pic_name)
#         print(total_lable, "total_pic:", total_pic, total_lable == total_pic)

# plt.ion()
# # fig = plt.figure("Pnet")
# # plt.plot([1, 2, 3], [4, 5, 6])
# # plt.title("12 23 34 45 dsfsdfsdfsdfsdfsdfsdfsdgdfhfdhfdhfgdg\nhfdhfgdhfdghfdghdffd", fontsize=8, color="blue")
# # plt.pause(10)
# # fig.savefig(r"E:\1234.jpg")
# # plt.clf()
# # plt.plot([1, 2, 3], [8, 6, 5])
# # plt.pause(2)
# # plt.ioff()
# # fig.savefig(r"E:\123.jpg")

# a = np.array(1.23445)
# print(a)
# b = np.set_printoptions(a, 2, 2)
# print(b)

# print(time.strftime('%Y%m%d%H%M%S'))

# y = torch.ones(1,requires_grad=True)
# print(y.requires_grad)
# y1 = y + 1
# print(y1.requires_grad)
# print(torch.__version__)
# with torch.no_grad():
#     y1 = y + 1
#     x = torch.ones(1,requires_grad=True)
#     print(x.requires_grad)
#     print(y1.requires_grad)


# a = [[2, 3, 1, 5, 4],[1,2,3]]
# b = a.copy()
# b.remove([1,2,3])
# print(a)
# print(b)
# b = np.array([0.3,0.4,0.5,0])
# time1 = time.time()
# print(np.max(b))
# time2 = time.time()
# print(time2 - time1)
#
# time1 = time.time()
# print(np.sum(b))
# time2 = time.time()
# print(time2 - time1)


# a = 1
# b = 1
# for i in range(5):
#     if a == 1:
#         if b == 0:
#             continue
#     print(a)


# a = np.array([1,2,3],dtype=np.int32)
# b = np.array([2,3,4],dtype=np.int32)
# print(np.true_divide(a,b))


# x = tf.ones(shape = [1,5,5,3])
# y = tf.random_uniform(shape = [1,5,5,3],minval= 0)
# w = tf.Variable(tf.truncated_normal(shape = [1, 3, 3, 1], dtype=tf.float32))
# conv = tf.nn.conv2d(x,w,[1,2,2,1],padding="SAME")
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(x).shape)
#     print(sess.run(conv).shape)

# print(ord('A'))
# y = [[0]*64 for _ in range(4)]
# print(y[3][63])

# e = tf.constant(1, dtype=tf.float32)
# a = tf.constant([0, 1, 1, 3, 4, 5], dtype=tf.float32)
# b = tf.map_fn(lambda x: tf.cast(tf.equal(x, 1), dtype=tf.float32), a)
# c = tf.reduce_mean(tf.map_fn(lambda x: tf.cast(tf.equal(x, 1), dtype=tf.float32), a))
# with tf.Session() as sess:
#     a, b, c, e = sess.run([a, b, c, e])
#     print(a, b, c, e)

# def softmax(x):
#     y = np.divide(np.exp(x), np.sum(np.exp(x)))
#     return y
#
# plt.ion()
# x = np.linspace(-1, 1, 100)
# y = softmax(x)
# print(x)
# print(y)
# plt.clf()
# plt.plot(x, y)
# plt.show()
# plt.pause(0)
# plt.ioff()

# arr2[1:2, 3:7] = 3
# s = slice(1, 11, 4)
# arr2 = np.arange(12)
# print(arr2)
# print(arr2[s])
# print(arr2[1:11:4])


# negative_anno_filename = r"C:\mywiderface_train\48\negativebak0629combine.txt"
# new_filename = r"C:\mywiderface_train\48\negative.txt"
# negative_anno_file = open(negative_anno_filename, "r")
# all_strs = negative_anno_file.readlines()
# f = open(new_filename, "a")
# # new_strs = []
# # for i in all_strs:
# #     # a = i.split()
# #     # a[1] = '0.9'
# #     new_strs.append(" ".join(a))
# # print(new_strs)
# # exit()
# # del all_strs[::3]
# # f.writelines("\n".join(new_strs))
# f.writelines(all_strs[::2])
# # f.writelines(all_strs)
# f.close()

# negative_anno_filename = r"C:\mywiderface_dev\48\negative.bak.txt"
# new_filename = r"C:\mywiderface_dev\48\negative.txt"
# negative_anno_file = open(negative_anno_filename, "r")
# all_strs = negative_anno_file.readlines()
# f = open(new_filename, "a")
# # new_strs = []
# # for i in all_strs:
# #     a = i.split()
# #     a[1] = '0.9'
# #     new_strs.append(" ".join(a))
# # f.writelines("\n".join(new_strs))
# f.writelines(all_strs[::2])
# f.close()

num = 0


def say(msg, j, lock):
    print("msg:{0}".format(msg))
    time.sleep(3 - msg/2)
    # print("end msg:{0}".format(msg))
    return msg
    # global num
    # while True:
    #     print(j, num)
    #     if j == num:
    #         # lock.acquire()
    #         print("end msg:{0}".format(msg))
    #         num += 1
    #         while True:
    #             print(num, "========")
    #         # lock.release()
    #         break
    #     else:
    #         continue


# def init(l):
#     global lock
#     lock = l


if __name__ == '__main__':
    # lock = Lock()
    #
    # t1 = Thread(target=say, args=())

    pool = mp.Pool(3)
    manager = Manager()
    l1 = manager.Lock()
    l2 = manager.Lock()
    l3 = manager.Lock()
    num = 0
    j = 0
    start_time = time.time()
    a = []
    for i in range(6):
        # pool.apply_async(say, args=(i, j, l1))
        # # pool.map(say, )
        # j += 1
        a.append(pool.apply_async(say, args=(i, num, l1)))
        if (i+1) % 3 == 0:
            print([a1.get() for a1 in a[-3:]])
        # pool.apply_async(say, args=(i, num,))
        # p1 = mp.Process(target=say, args=(i, num, l))
        # p2 = mp.Process(target=say, args=(i, num, l))
        # p3 = mp.Process(target=say, args=(i, num, l))
        # p1.start()
        # p2.start()
        # p3.start()
        # p1.join()
        # p2.join()
        # p3.join()
    pool.close()
    pool.join()
    print("=========")
    print("end")
    print(time.time() - start_time)
    # for a_ in a:
    #     print(a_.get())



