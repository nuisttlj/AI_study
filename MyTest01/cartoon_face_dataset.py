import os
import tensorflow as tf
import numpy as np
from PIL import Image


class MyDataset:
    def __init__(self, path, batch_num):
        self.path = path
        self.filenames = os.listdir(self.path)
        self.filenames = tf.constant(self.filenames)
        self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        self.dataset = self.dataset.map(
            lambda filename: tuple(
                tf.py_func(self._read_by_function, [filename],
                           [tf.float32])))
        self.dataset = self.dataset.shuffle(buffer_size=1000)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(batch_num)

        iterator = self.dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()

    def get_batch(self, sess):
        return sess.run(self.next_element)

    def _read_by_function(self, filename):
        # print(type(filename))
        _filename = bytes.decode(filename)
        pic_path = os.path.join(self.path, _filename)
        pic = Image.open(pic_path)
        pic_data = (np.array(pic, dtype=np.float32) / 255 - 0.5) * 2
        return pic_data


if __name__ == '__main__':
    mydataset = MyDataset(r"C:\faces", 1)
    with tf.Session() as sess:
        xs = mydataset.get_batch(sess)
        print(xs[0].shape)
