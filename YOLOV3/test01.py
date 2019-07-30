import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# print(np.log(3))
# print(np.max((2, 3)))
# print(list(map(float, [1, 2, 3])))
# print(np.array(['3', '2'], dtype=np.float32))
# print(np.split(np.array([1, 2, 3]), 3))

# IMAGE_BASE_DIR = r'/home/coco/train2014'
# LABEL_FILE_PATH = r'/home/coco/labels/train2014'
# name_list = os.listdir(LABEL_FILE_PATH)
# name_list.sort()
# f_new = open(r'/home/coco/labels/train2014test500.txt', "w")
# j = 0
# for i, name in enumerate(name_list):
#     if i > 0:
#         if j < 500:
#             if np.array(Image.open(os.path.join(IMAGE_BASE_DIR, name.replace('.txt', '.jpg')))).shape[-1] != 3:
#                 continue
#             elif len(np.array(Image.open(os.path.join(IMAGE_BASE_DIR, name.replace('.txt', '.jpg')))).shape) != 3:
#                 continue
#             else:
#                 f = open(os.path.join(LABEL_FILE_PATH, name))
#                 line = f.readlines()
#                 new_line = [x for strs in line for x in strs.split()]
#                 if i > 1:
#                     f_new.write("\n")
#                 f_new.write(name.replace('.txt', '.jpg')+' ')
#                 f_new.writelines(" ".join(new_line))
#                 j += 1

# a = torch.Tensor([[1, 2], [3, 4], [5, 6]])
# b = torch.Tensor([[1], [3], [5]])
# c = torch.Tensor([[7], [8], [9]])
# print(torch.cat((c, a, b), dim=1))

# input = torch.randn(3, 5, requires_grad=True)
# # target = torch.randint(5, (3,), dtype=torch.int64)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(input, target)

# a = np.asarray([[[4, 7], [5, 8]], [[1, 3], [2, 4]]])
# # b = np.argmax(a, axis=1)
# print(a[..., 0])
# c = np.argmax(a[..., 0], axis=1)
# # print(b)
# # print(a[b[0]][b[1]])
# print(c)
# print(a[c])

# a = np.array([[1, 22, 3, 4], [1, 33, 5, 6], [3, 44, 5, 6], [2, 44, 56, 6]])
#
# print(np.unique(a[:, 0]))
# list_a = []
# for cls in np.unique(a[:, 0]):
#     idxs = a[:, 0] == cls
#     b = a[idxs]
#     list_a.append(b)
# print(list_a)
# print(np.concatenate(list_a, axis=0))


# print(torch.tensor(0.))
# print(torch.arange(3))
#
# torch.index_select()

# fig, ax = plt.subplots()
# # # ncolors = len(plt.rcParams['axes.prop_cycle'])
# a = np.random.randint(1, 10, (10, 2))
# for i in range(10):
#     plt.ion()
#     ax.scatter(a[i][0], a[i][1])
#     plt.show()
#     plt.pause(1)
#     plt.ioff()

# b = np.random.randint(1, 10, (10,))
# print(a)
# print(b)
# print(a[b==3])

# print(torch.__version__)


# w = torch.randn((2, 10))
# w_normed = F.normalize(w, dim=0)/10
# w_norm_value = torch.norm(w_normed, dim=0)
# print(w)
# print(w_normed)
# print(w_norm_value)

inputs = torch.randn(1, 4, 5, 5)
weights = torch.randn(4, 8, 3, 3)
# out = F.conv_transpose2d(inputs, weights, padding=1)
out = nn.ConvTranspose2d(4, 8, 3)(inputs)
print(out.size())

# w = torch.empty(1, 2, 3, 3)
# print(w)
# torch.nn.init.kaiming_normal_(w)
# print(w)


# img_name = os.listdir(r"/home/Unet/data")
# img_name = [int(i.replace(".png", "")) for i in img_name]
# img_name.sort()
# label_name = os.listdir(r"/home/Unet/label")
# label_name.sort()
#
# print(len(img_name))
# print(len(label_name))
# print(img_name)
# print(label_name)

# w = ["{0}.png".format(i) for i in range(6247)]
#
# print(w)