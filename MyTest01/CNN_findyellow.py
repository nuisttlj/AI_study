import tensorflow as tf
import matplotlib.pyplot as plt
import make_picdataset
import numpy as np
import os

mydata = make_picdataset.MyDataset("compounds_pic", 16)


class CNNNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])
        self.offset_y = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        self.conf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=np.sqrt(2 / (3*3*3))))
        self.conv1_b = tf.Variable(tf.zeros(64))

        self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=np.sqrt(2 / (3*3*64))))
        self.conv2_b = tf.Variable(tf.zeros(128))

        self.conv3_w = tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=np.sqrt(2 / (3*3*128))))
        self.conv3_b = tf.Variable(tf.zeros(64))

        self.in1_w = tf.Variable(tf.truncated_normal([32 * 32 * 64, 400], stddev=tf.sqrt(2 / (32*32*64))))
        self.in1_b = tf.Variable(tf.zeros(400))

        self.in2_w = tf.Variable(tf.truncated_normal([400, 200], stddev=tf.sqrt(2 / 400)))
        self.in2_b = tf.Variable(tf.zeros(200))

        self.out1_w = tf.Variable(tf.truncated_normal([200, 4], stddev=tf.sqrt(1 / 200)))
        self.out1_b = tf.Variable(tf.zeros(4))

        self.out2_w = tf.Variable(tf.truncated_normal([200, 1], stddev=tf.sqrt(1 / 200)))
        self.out2_b = tf.Variable(tf.zeros(1))

    def forward(self):
        conv1 = tf.nn.conv2d(self.x, self.conv1_w, [1, 1, 1, 1], padding="SAME") + self.conv1_b
        batch1 = tf.layers.batch_normalization(conv1)
        relu_conv1 = tf.nn.relu(batch1)
        pool1 = tf.nn.max_pool(relu_conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        conv2 = tf.nn.conv2d(pool1, self.conv2_w, [1, 1, 1, 1], padding="SAME") + self.conv2_b
        batch2 = tf.layers.batch_normalization(conv2)
        relu_conv2 = tf.nn.relu(batch2)
        pool2 = tf.nn.max_pool(relu_conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        conv3 = tf.nn.conv2d(pool2, self.conv3_w, [1, 1, 1, 1], padding="SAME") + self.conv3_b
        batch3 = tf.layers.batch_normalization(conv3)
        relu_conv3 = tf.nn.relu(batch3)
        pool3 = tf.nn.max_pool(relu_conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        flat = tf.reshape(pool3, [-1, 32 * 32 * 64])

        fc1 = tf.matmul(flat, self.in1_w) + self.in1_b
        batch4 = tf.layers.batch_normalization(fc1)
        relu_fc1 = tf.nn.relu(batch4)

        fc2 = tf.matmul(relu_fc1, self.in2_w) + self.in2_b
        batch5 = tf.layers.batch_normalization(fc2)
        relu_fc2 = tf.nn.relu(batch5)

        self.output1 = tf.matmul(relu_fc2, self.out1_w) + self.out1_b
        self.output2 = tf.matmul(relu_fc2, self.out2_w) + self.out2_b
        self.conf_output = tf.nn.sigmoid(self.output2)

    def loss(self):
        self.offset_mask = tf.where(tf.reshape(self.conf_y, [-1]) > 0)
        self.offset_loss = tf.reduce_mean(
            (tf.gather_nd(self.output1, self.offset_mask) - tf.gather_nd(self.offset_y, self.offset_mask)) ** 2)
        self.conf_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.conf_y, logits=self.output2))
        self.loss = self.offset_loss + self.conf_loss

    def backword(self):
        self.op = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    net = CNNNet()
    net.forward()
    net.loss()
    net.backword()
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    if os.path.exists("./yellow_param/checkpoint"):
        saver.restore(sess, "./yellow_param/findyellow_model.ckpt")

    # plt.ion()
    # plt_x, plt_y = [], []
    for i in range(1000):
        xs, off_ys, conf_ys = mydata.get_batch(sess)
        # print(xs.shape, ys.shape)
        _loss, _offset_loss, _conf_loss, _, _offset_output, _conf_output = sess.run(
            [net.loss, net.offset_loss, net.conf_loss, net.op, net.output1, net.conf_output],
            feed_dict={net.x: xs, net.offset_y: off_ys,
                       net.conf_y: conf_ys})

        # print(type(_loss))
        # print(_output[0])
        # print(ys[0])
        # plt_x.append(i)
        # plt_y.append(_loss)
        # plt.clf()
        # plt.plot(plt_x, plt_y)
        saver.save(sess, "./yellow_param/findyellow_model.ckpt")
        # if i % 5 == 0:
        print("training_num:", i + 1)
        print("total loss:", _loss, " offset loss:", _offset_loss, " confidence loss:", _conf_loss)
        print(_offset_output[0])
        print(off_ys[0])
        print(_conf_output[0])
        print(conf_ys[0])
        # exit()
        # saver.save(sess, "./yellow_param/findyellow_model.ckpt")
        # print(_output)
        # print(net.y.shape)
    # plt.ioff()
