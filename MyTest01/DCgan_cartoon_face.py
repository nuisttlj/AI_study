import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cartoon_face_dataset import MyDataset
import os


class D_Net:
    def __init__(self):
        with tf.variable_scope("d_net"):
            self.w1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[5, 5, 3, 64], stddev=0.02))
            self.b1 = tf.Variable(tf.zeros([64]), dtype=tf.float32)
            self.w2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[5, 5, 64, 128], stddev=0.02))
            self.b2 = tf.Variable(tf.zeros([128]), dtype=tf.float32)
            self.w3 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[5, 5, 128, 256], stddev=0.02))
            self.b3 = tf.Variable(tf.zeros([256]), dtype=tf.float32)
            self.w4 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[5, 5, 256, 512], stddev=0.02))
            self.b4 = tf.Variable(tf.zeros([512]), dtype=tf.float32)
            self.w5 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[6 * 6 * 512, 1], stddev=0.02))
            self.b5 = tf.Variable(tf.zeros([1]), dtype=tf.float32)

    def forward(self, x, reuse=False):
        with tf.variable_scope("d_net", reuse=reuse):
            net = tf.nn.leaky_relu(tf.nn.conv2d(x, self.w1, [1, 2, 2, 1], padding="SAME") + self.b1)
            net = tf.nn.leaky_relu(
                tf.layers.batch_normalization(tf.nn.conv2d(net, self.w2, [1, 2, 2, 1], padding="SAME") + self.b2,
                                              momentum=0.9, epsilon=1e-5))
            net = tf.nn.leaky_relu(
                tf.layers.batch_normalization(tf.nn.conv2d(net, self.w3, [1, 2, 2, 1], padding="SAME") + self.b3,
                                              momentum=0.9, epsilon=1e-5))
            net = tf.nn.leaky_relu(
                tf.layers.batch_normalization(tf.nn.conv2d(net, self.w4, [1, 2, 2, 1], padding="SAME") + self.b4,
                                              momentum=0.9, epsilon=1e-5))
            net = tf.matmul(tf.reshape(net, [128, 6 * 6 * 512]), self.w5) + self.b5
            params = tf.trainable_variables()
            self.params = [var for var in params if 'd_net' in var.name]
            # print(self.params)
            # exit()
            return net


class G_Net:
    def __init__(self):
        with tf.variable_scope("g_net"):
            self.w1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[128, 6 * 6 * 512], stddev=0.02))
            self.b1 = tf.Variable(tf.zeros([6 * 6 * 512]), dtype=tf.float32)
            self.w2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[5, 5, 256, 512], stddev=0.02))
            self.b2 = tf.Variable(tf.zeros([256]), dtype=tf.float32)
            self.w3 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[5, 5, 128, 256], stddev=0.02))
            self.b3 = tf.Variable(tf.zeros([128]), dtype=tf.float32)
            self.w4 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[5, 5, 64, 128], stddev=0.02))
            self.b4 = tf.Variable(tf.zeros([64]), dtype=tf.float32)
            self.w5 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[5, 5, 3, 64], stddev=0.02))
            self.b5 = tf.Variable(tf.zeros([3]), dtype=tf.float32)

    def forward(self, x, reuse=False):
        with tf.variable_scope("g_net", reuse=reuse):
            net = tf.matmul(x, self.w1) + self.b1
            net = tf.nn.relu(
                tf.layers.batch_normalization(tf.reshape(net, [-1, 6, 6, 512]), momentum=0.9, epsilon=1e-5))
            net = tf.nn.relu(
                tf.layers.batch_normalization(
                    tf.nn.conv2d_transpose(net, self.w2, [128, 12, 12, 256], [1, 2, 2, 1], padding="SAME") + self.b2,
                    momentum=0.9, epsilon=1e-5))
            net = tf.nn.relu(
                tf.layers.batch_normalization(
                    tf.nn.conv2d_transpose(net, self.w3, [128, 24, 24, 128], [1, 2, 2, 1], padding="SAME") + self.b3,
                    momentum=0.9, epsilon=1e-5))
            net = tf.nn.relu(
                tf.layers.batch_normalization(
                    tf.nn.conv2d_transpose(net, self.w4, [128, 48, 48, 64], [1, 2, 2, 1], padding="SAME") + self.b4,
                    momentum=0.9, epsilon=1e-5))
            net = tf.nn.tanh(
                tf.nn.conv2d_transpose(net, self.w5, [128, 96, 96, 3], [1, 2, 2, 1], padding="SAME") + self.b5)
            params = tf.trainable_variables()
            self.params = [var for var in params if 'g_net' in var.name]
            # print(self.params)
            return net


class Net:
    def __init__(self):
        self.x = tf.placeholder(shape=[None, 96, 96, 3], dtype=tf.float32)
        self.init_data = tf.placeholder(shape=[None, 128], dtype=tf.float32)
        self.fake_label = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.real_label = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    def forward(self):
        self.d_net = D_Net()
        self.g_net = G_Net()
        self.g_out = self.g_net.forward(self.init_data)
        self.d_real_out = self.d_net.forward(self.x)
        self.d_fake_out = self.d_net.forward(self.g_out, reuse=True)

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


def visit_image(batchsize, samples, i):
    fig, axes = plt.subplots(figsize=(48, 48), nrows=8, ncols=16, sharex=True, sharey=True)
    for ax, img in zip(axes.flatten(), samples[-1]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((96, 96, 3)))
    plt.savefig(r'C:\gan_faces\{0}.jpg'.format(i))


if __name__ == '__main__':
    net = Net()
    net.forward()
    net.loss()
    net.backward()
    init = tf.global_variables_initializer()
    mydataset = MyDataset(r"C:\faces", 128)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(init)
        # num = os.listdir("./gen_face/")[-1].split(".")[0]
        # if os.path.exists("./gen_face/{0}/checkpoint".format(num)):
        #     saver.restore(sess, "./gen_face/{0}/gen_face.ckpt".format(num))
        # else:
        #     sess.run(init)
        for i in range(100000):
            xs = mydataset.get_batch(sess)[0]
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
            for _ in range(2):
                g_loss_, _ = sess.run([net.g_loss, net.bp_g],
                                      feed_dict={net.init_data: init_datas, net.fake_label: g_fake_labels})
                print("G_loss: ", g_loss_)
            if i % 50 == 0:
                # init_datas = np.random.normal(0, 0.02, (128, 128))
                init_datas = np.random.uniform(-1, 1, (128, 128))
                g_out_ = sess.run(net.g_out, feed_dict={net.init_data: init_datas})
                img_array = np.array(g_out_) / 2 + 0.5
                # plt.imshow(img_array)
                # plt.pause(0.1)

                visit_image(-1, [img_array], i)

            if i % 50 == 0:
                saver.save(sess, "./gen_face/{0}/gen_face.ckpt".format(i))
