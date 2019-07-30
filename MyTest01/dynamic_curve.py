import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import PIL.Image as image

# x = np.linspace(0, 2 * np.pi, 100)
# y1, y2 = np.sin(x), np.cos(x)
# plt.title("sin&cos")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.show()

# x, y = [], []
# plt.ion()
# for i in range(100):
#     x.append(i)
#     y.append(i**2)
#     plt.clf()
#     plt.plot(x,y)
#     plt.pause(0.3)
# plt.ioff()


x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)

image = plt.figure()
three_d = Axes3D(image)
three_d.scatter(x, y, z)
plt.show()

# name_list = ["A", "B", "C", "D"]
# value_list1 = [1.3, 2.2, 5.6, 8.9]
# value_list2 = [3, 5, 2, 7]
# # plt.bar(range(len(name_list)),value_list1,tick_label = name_list,color = 'rgby')
# # # plt.show()
# x = list(range(len(name_list)))
# width = 0.3
# plt.bar(x, value_list1, tick_label=name_list, width=width, label="boy", fc="b")
# for i in x:
#     x[i] += width
# plt.bar(x, value_list2, tick_label=name_list, width=width, label="girl", fc="r")
# plt.legend()
# plt.show()

# pic = plt.imread("pic3.jpg")
# # pic = image.open("pic3.jpg")
# plt.axis("off")
# plt.imshow(pic)
# plt.show()

# im = image.open("pic3.jpg")
# print(list(im.load()[12, 14]))
# im_data = np.array(im)
# print(im_data[12, 14])
