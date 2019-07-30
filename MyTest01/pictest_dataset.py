import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class MyDataset:
    def __init__(self, path, batch_num):
        self.path = path
        self.filenames = os.listdir(self.path)
        self.labels = list(map(lambda filename: list(map(float, filename.split(".")[1:6])), self.filenames))
        # print(self.labels)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
        # print(self.dataset)
        # print(type(self.filenames[0]),type(self.labels[0]))
        self.dataset = self.dataset.map(
            lambda filename, label: tuple(tf.py_func(self._read_by_function, [filename, label],
                                                     [tf.float32, label.dtype])))
        # self.dataset = self.dataset.shuffle(buffer_size=100)
        # self.dataset = self.dataset.repeat()
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

    def _read_by_function(self, filename, label):
        # print(type(filename))
        _filename = bytes.decode(filename)
        pic_path = os.path.join(self.path, _filename)
        pic = Image.open(pic_path)
        pic_data = np.array(pic, dtype=np.float32) / 255 - 0.5
        label = label / 255 - 0.5
        return pic_data, label


if __name__ == "__main__":
    mydata = MyDataset("compounds_pic_test", 1)
    # print(mydata.filenames)
    # print(mydata.labels)
    # print(mydata.dataset)
    with tf.Session() as sess:
        xs, ys = mydata.get_batch(sess)
        print(ys)
        # print(ys)
        # print(xs.shape, ys.shape)
        xs, ys = mydata.get_batch(sess)
        # print(xs.shape, ys.shape)
        # print(ys)

