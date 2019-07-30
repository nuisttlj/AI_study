import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


class MyDataset:
    def __init__(self, path, batch_num):
        self.path = path
        self.filenames = os.listdir(self.path)
        self.off_labels = list(map(lambda filename: list(map(float, filename.split(".")[1:5])), self.filenames))
        self.conf_labels = list(map(lambda filename: list(map(float, filename.split(".")[5:6])), self.filenames))
        self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.off_labels, self.conf_labels))
        # print(self.dataset)
        # print(type(self.filenames[0]),type(self.labels[0]))
        self.dataset = self.dataset.map(
            lambda filename, off_label, conf_label: tuple(
                tf.py_func(self._read_by_function, [filename, off_label, conf_label],
                           [tf.float32, off_label.dtype, conf_label.dtype])))
        # print(self.dataset)
        self.dataset = self.dataset.shuffle(buffer_size=1000)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(batch_num)

        iterator = self.dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()

    def get_batch(self, sess):
        # self.dataset = self.dataset.shuffle(buffer_size=100)
        # self.dataset = self.dataset.repeat()
        # self.dataset = self.dataset.batch(batch_num)
        #
        # iterator = self.dataset.make_one_shot_iterator()
        # self.next_element = iterator.get_next()
        return sess.run(self.next_element)

    def _read_by_function(self, filename, off_label, conf_label):
        # print(type(filename))
        _filename = bytes.decode(filename)
        pic_path = os.path.join(self.path, _filename)
        pic = Image.open(pic_path)
        pic_data = np.array(pic, dtype=np.float32) / 255 - 0.5
        off_label = off_label / 256
        return pic_data, off_label, conf_label


if __name__ == "__main__":
    mydata = MyDataset("compounds_pic", 20)
    # print(mydata.filenames)
    # print(mydata.labels)
    # print(mydata.dataset)
    with tf.Session() as sess:
        for i in range(2):
            xs, off_ys, conf_ys = mydata.get_batch(sess)
            # print(xs[1])
            print(xs[1].shape)
            print(off_ys)
            print(conf_ys)
            # print(ys)
            # print(xs.shape, ys.shape)
            # xs, ys = mydata.get_batch(sess)
            # print(xs.shape, ys.shape)
            # print(ys)
