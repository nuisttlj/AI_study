import tensorflow as tf
import pictest_dataset
import matplotlib.pyplot as plt
import cv2
import make_picdataset
import numpy as np
import os

# mydata = pictest_dataset.MyDataset("compounds_pic_test", 1)
mydata = make_picdataset.MyDataset("compounds_pic_test", 100)


class CNNNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])
        self.offset_y = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        self.conf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=np.sqrt(2 / (3*3*3))))
        self.conv1_b = tf.Variable(tf.zeros(64))

        self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=np.sqrt(2 / (3*3*64))))
        self.conv2_b = tf.Variable(tf.zeros(128))

        self.conv3_w = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=np.sqrt(2 / (3*3*128))))
        self.conv3_b = tf.Variable(tf.zeros(128))

        self.conv4_w = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=np.sqrt(2 / (3*3*128))))
        self.conv4_b = tf.Variable(tf.zeros(256))

        self.conv5_w = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=np.sqrt(2 / (3*3*256))))
        self.conv5_b = tf.Variable(tf.zeros(256))

        self.in1_w = tf.Variable(tf.truncated_normal([8 * 8 * 256, 800], stddev=tf.sqrt(2 / (8*8*256))))
        self.in1_b = tf.Variable(tf.zeros(800))

        self.in2_w = tf.Variable(tf.truncated_normal([800, 200], stddev=tf.sqrt(2 / 800)))
        self.in2_b = tf.Variable(tf.zeros(200))

        self.out1_w = tf.Variable(tf.truncated_normal([200, 4], stddev=tf.sqrt(1 / 200)))
        self.out1_b = tf.Variable(tf.zeros(4))

        self.out2_w = tf.Variable(tf.truncated_normal([200, 1], stddev=tf.sqrt(1 / 200)))
        self.out2_b = tf.Variable(tf.zeros(1))

    def forward(self):
        conv1 = tf.nn.conv2d(self.x, self.conv1_w, [1, 1, 1, 1], padding="SAME") + self.conv1_b
        # batch1 = tf.layers.batch_normalization(conv1)
        relu_conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(relu_conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        conv2 = tf.nn.conv2d(pool1, self.conv2_w, [1, 1, 1, 1], padding="SAME") + self.conv2_b
        # batch2 = tf.layers.batch_normalization(conv2)
        relu_conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(relu_conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        conv3 = tf.nn.conv2d(pool2, self.conv3_w, [1, 1, 1, 1], padding="SAME") + self.conv3_b
        # batch3 = tf.layers.batch_normalization(conv3)
        relu_conv3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(relu_conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        conv4 = tf.nn.conv2d(pool3, self.conv4_w, [1, 1, 1, 1], padding="SAME") + self.conv4_b
        # batch4 = tf.layers.batch_normalization(conv4)
        relu_conv4 = tf.nn.relu(conv4)
        pool4 = tf.nn.max_pool(relu_conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        conv5 = tf.nn.conv2d(pool4, self.conv5_w, [1, 1, 1, 1], padding="SAME") + self.conv5_b
        # batch5 = tf.layers.batch_normalization(conv5)
        relu_conv5 = tf.nn.relu(conv5)
        pool5 = tf.nn.max_pool(relu_conv5, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        flat = tf.reshape(pool5, [-1, 8 * 8 * 256])

        fc1 = tf.matmul(flat, self.in1_w) + self.in1_b
        # batch6 = tf.layers.batch_normalization(fc1)
        relu_fc1 = tf.nn.relu(fc1)

        fc2 = tf.matmul(relu_fc1, self.in2_w) + self.in2_b
        # batch7 = tf.layers.batch_normalization(fc2)
        relu_fc2 = tf.nn.relu(fc2)

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
    # sess.run(init)
    saver = tf.train.Saver()
    if os.path.exists("./yellow_param02/checkpoint"):
        saver.restore(sess, "./yellow_param02/findyellow_model.ckpt")

    xs, off_ys, conf_ys = mydata.get_batch(sess)
    _offset_output, _conf_output = sess.run([net.output1, net.conf_output], feed_dict={net.x: xs, net.offset_y: off_ys,
                                                                                       net.conf_y: conf_ys})
    # print(_conf_output)
    # exit(0)
    # print(ys)
    for _x, i, j in zip(xs, _offset_output, _conf_output):
        # print(i , j, j > 0.9)
        # print(_x.shape)
        # exit()
        plt.ion()
        pic = _x + 0.5
        if j > 0.9:
            offset = i * 256
            cv2.rectangle(pic, (offset[0], offset[1]), (offset[2], offset[3]), (0, 0, 255), 1)
        plt.imshow(pic)
        plt.pause(2)
        plt.ioff()

