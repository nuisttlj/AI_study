import os
import numpy as np
import PIL.Image as image
import random as ran
import cv2

dir = "bgpic"
x = 1

for filename in os.listdir(dir):
    empty_bgpic = image.open("{0}\{1}".format(dir, filename))
    empty_bgpic_convert = empty_bgpic.convert("RGB")
    # shape = np.shape(empty_bgpic)
    # if len(shape) == 3 and shape[2] == 3:
    #         empty_bgpic = empty_bgpic
    #     else:
    #         continue

    empty_bgpic_resize = empty_bgpic_convert.resize((256, 256))
    empty_bgpic_resize.save("compounds_pic\{0}{1}.png"
                            .format(x, ".0.0.0.0.0"))
    x += 1
    # if x == 500:
    #     break
