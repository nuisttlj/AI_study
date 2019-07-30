import tensorflow as tf
import os
from PIL import Image
import numpy as np


class FaceDataset:
    def __init__(self, path, batch_num, epoch):
        self.image_path = path
        self.sample_data = []
        self.sample_data.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.sample_data.extend(open(os.path.join(path, "part.txt")).readlines())
        self.sample_data.extend(open(os.path.join(path, "negative.txt")).readlines())
        # print(len(self.sample_data))
        self.image_filenames = []
        self.image_lables = []
        for sampledata in self.sample_data:
            self.image_filenames.append(sampledata.strip().split()[0])
            self.image_lables.append(list(map(float, sampledata.strip().split()[1:])))
        self.dataset = tf.data.Dataset.from_tensor_slices((self.image_filenames, self.image_lables))
        self.dataset = self.dataset.map(
            lambda image_filename, image_lable: tuple(
                tf.py_func(self._read_by_func, [image_filename, image_lable], [tf.float32, tf.float32])))
        self.dataset = self.dataset.shuffle(buffer_size=2000).repeat(epoch).batch(batch_num)
        iterator = self.dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()

    def get_batch(self, sess):
        return sess.run(self.next_element)

    def _read_by_func(self, filename, lable):
        _filename = bytes.decode(filename)
        path_filename = os.path.join(self.image_path, _filename)
        image = Image.open(path_filename)
        image_data = np.array(image, dtype=np.float32) / 255 - 0.5
        image_lable = np.array(lable, dtype=np.float32)
        return image_data, image_lable


if __name__ == '__main__':
    my_facedata = FaceDataset(path=r"E:\myceleba\12", batch_num=512, epoch=15)
    with tf.Session() as sess:
        xs, ys = my_facedata.get_batch(sess=sess)
        print(xs[0])
        print(xs[0].shape)
        print(ys[0])
