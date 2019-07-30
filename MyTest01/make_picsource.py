import os
import numpy as np
import PIL.Image as image
import random as ran
import cv2

dir = "bgpic"
x = 1

for filename in os.listdir(dir):
    # print(filename)
    bgpic = image.open("{0}\{1}".format(dir, filename))
    bgpic_convert = bgpic.convert("RGB")
    # shape = np.shape(bgpic)
    # print(shape)
    # if len(shape) == 3 and shape[2] == 3:
    #     bgpic = bgpic
    # else:
    #     continue
    bgpic_resize = bgpic_convert.resize((256, 256))
    # print(np.shape(bgpic_resize))

    name = np.random.randint(1, 21)
    yellow = image.open("yellow\{0}.png".format(name))

    yellow = yellow.rotate(np.random.randint(-45, 45))
    ran_w = np.random.randint(30, 100)
    ran_h = ran_w
    yellow_new = yellow.resize((ran_w, ran_h))

    ran_x1 = np.random.randint(0, 256 - ran_w)
    ran_y1 = np.random.randint(0, 256 - ran_h)

    r, g, b, a = yellow_new.split()
    bgpic_resize.paste(yellow_new, (ran_x1, ran_y1), mask=a)

    ran_x2 = ran_x1 + ran_w
    ran_y2 = ran_y1 + ran_h

    bgpic_resize.save("compounds_pic/{0}{1}.png"
                      .format(x, "." + str(ran_x1) + "." + str(ran_y1) + "." + str(ran_x2) + "."
                              + str(ran_y2) + "." + "1"))
    x += 1
    # if x == 1000:
    #     break
