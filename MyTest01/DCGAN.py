import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
import numpy as np
import matplotlib.pyplot as plt


class D_Net:
    # def __init__(self):
    #     self.w1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[3, 3, 1, 64], stddev=0.02))
    #     self.b1 = tf.Variable(tf.zeros([64]))
    #     self.w2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[3, 3, 64, 64], stddev=0.02))
    #     self.b2 = tf.Variable(tf.zeros([64]))
    #     self.w3 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[3, 3, 64, 128], stddev=0.02))
    #     self.b3 = tf.Variable(tf.zeros([128]))
    #     self.w4 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[3, 3, 128, 128], stddev=0.02))
    #     self.b4 = tf.Variable(tf.zeros([128]))
    #     self.w5 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[7, 7, 128, 1], stddev=0.02))
    #     self.b5 = tf.Variable(tf.zeros([1]))
    #
    #
    # def forward(self, x):
    #     net = tf.nn.leaky_relu(tf.nn.conv2d(x, self.w1, [1, 1, 1, 1], padding="SAME") + self.b1)
    #     net = tf.nn.leaky_relu(
    #         tf.layers.batch_normalization(tf.nn.conv2d(net, self.w2, [1, 2, 2, 1], padding="SAME") + self.b2))
    #     net = tf.nn.leaky_relu(
    #         tf.layers.batch_normalization(tf.nn.conv2d(net, self.w3, [1, 1, 1, 1], padding="SAME") + self.b3))
    #     net = tf.nn.leaky_relu(
    #         tf.layers.batch_normalization(tf.nn.conv2d(net, self.w4, [1, 2, 2, 1], padding="SAME") + self.b4))
    #     net = tf.nn.leaky_relu(
    #         tf.layers.batch_normalization(tf.nn.conv2d(net, self.w5, [1, 1, 1, 1], padding="VALID") + self.b5))
    #     self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5, self.b5]
    #     return net

    def forward(self, x, reuse=False):
        with tf.variable_scope("d_net", reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding="SAME",
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                net = slim.conv2d(x, 64, 5, 2)
                net = tf.nn.leaky_relu(net)
                net = slim.conv2d(net, 128, 5, 2)
                net = tf.nn.leaky_relu(slim.batch_norm(net, decay=0.9, epsilon=1e-5))
                net = slim.fully_connected(tf.reshape(net, [128, 7 * 7 * 128]), 1)
                self.params = slim.get_model_variables(scope="d_net")
                # print('D_net', self.params)
                # exit()
        return net


class G_Net:
    def forward(self, x):
        with tf.variable_scope("g_net"):
            net = slim.fully_connected(x, 7 * 7 * 128)
            net = tf.nn.relu(slim.batch_norm(tf.reshape(net, [-1, 7, 7, 128]), decay=0.9, epsilon=1e-5))
            with slim.arg_scope([slim.conv2d_transpose],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                net = slim.conv2d_transpose(net, 64, 5, 2)
                net = tf.nn.relu(slim.batch_norm(net, decay=0.9, epsilon=1e-5))
                net = slim.conv2d_transpose(net, 1, 5, 2, activation_fn=tf.nn.tanh, normalizer_fn=None)
                net = tf.nn.tanh(net)
                self.params = slim.get_model_variables(scope="g_net")
                # print('G_net', self.params)
        return net


class Net:
    def __init__(self):
        self.x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
        self.init_data = tf.placeholder(shape=[None, 128], dtype=tf.float32)
        self.fake_label = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.real_label = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    def forward(self):
        x = tf.reshape(self.x, [-1, 28, 28, 1])
        self.d_net = D_Net()
        self.g_net = G_Net()
        self.g_out = self.g_net.forward(self.init_data)
        self.d_real_out = self.d_net.forward(x)
        self.d_fake_out = self.d_net.forward(self.g_out, reuse=True)
        # total_d_input = tf.concat([x, self.g_out], 0)
        # self.d_real_out, self.d_fake_out = self.d_net.forward(total_d_input)
        # print(self.d_real_out.shape, self.d_fake_out.shape)
        # exit()

    def loss(self):
        d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.real_label, logits=self.d_real_out))
        d_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.fake_label, logits=self.d_fake_out))
        self.d_loss = d_fake_loss + d_real_loss
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.fake_label, logits=self.d_fake_out))

    def backward(self):
        self.bp_g = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_net.params)
        self.bp_d = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_net.params)


if __name__ == '__main__':
    net = Net()
    net.forward()
    net.loss()
    net.backward()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(100000):
            xs, _ = mnist.train.next_batch(128)
            xs = (xs - 0.5) * 2
            # init_datas = np.random.normal(0, 0.02, (128, 128))
            init_datas = np.random.uniform(-1, 1, (128, 128))
            d_real_labels = np.ones(shape=[128, 1])
            d_fake_labels = np.zeros(shape=[128, 1])
            d_loss_, _ = sess.run([net.d_loss, net.bp_d],
                                  feed_dict={net.x: xs, net.init_data: init_datas, net.real_label: d_real_labels,
                                             net.fake_label: d_fake_labels})
            print("D_loss: ", d_loss_)
            # init_datas = np.random.normal(0, 0.02, (128, 128))
            g_fake_labels = np.ones(shape=[128, 1])
            g_loss_, _ = sess.run([net.g_loss, net.bp_g],
                                  feed_dict={net.init_data: init_datas, net.fake_label: g_fake_labels})
            print("G_loss: ", g_loss_)
            if i % 10 == 0:
                # init_datas = np.random.normal(0, 0.02, (128, 128))
                g_out_ = sess.run(net.g_out, feed_dict={net.init_data: init_datas})
                # print(np.array(g_out_).shape)
                # exit()
                img_array = np.array(g_out_)[0].reshape([28, 28]) / 2 + 0.5
                plt.imshow(img_array)
                plt.pause(0.1)
