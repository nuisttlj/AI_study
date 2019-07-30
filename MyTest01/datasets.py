import os
import tensorflow as tf
import numpy as np
from PIL import Image


class MyDataset:
    def __init__(self, path, batch_num):
        self.path = path
        self.filenames = os.listdir(self.path)
        self.labels = [self.__one_hot(filename.split(".")[0]) for filename in self.filenames]
        self.filenames = tf.constant(self.filenames)
        self.labels = tf.constant(self.labels, dtype=tf.float32)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
        # print(type(self.filenames[0]),type(self.labels[0]))
        self.dataset = self.dataset.map(
            lambda filename, label: tuple(
                tf.py_func(self._read_by_function, [filename, label],
                           [tf.float32, label.dtype])))
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

    def _read_by_function(self, filename, label):
        # print(type(filename))
        _filename = bytes.decode(filename)
        pic_path = os.path.join(self.path, _filename)
        pic = Image.open(pic_path)
        pic_data = np.array(pic, dtype=np.float32) / 255 - 0.5
        return pic_data, label

    def __one_hot(self, x):
        # y = np.zeros(shape=[4, 62], dtype=np.float32)
        y = [[0] * 62 for _ in range(6)]
        # y = [[0] * 10 for _ in range(4)]
        for i in range(len(x)):
            if ord(x[i]) <= 57:
                index = int(x[i])
            elif ord(x[i]) <= 90:
                index = ord(x[i]) - 55
            else:
                index = ord(x[i]) - 61
            y[i][index] = 1
        return y


if __name__ == '__main__':
    mydataset = MyDataset(r"D:\vertification", 1)
    with tf.Session() as sess:
        xs, ys = mydataset.get_batch(sess)
        print(xs)
        print(ys)
