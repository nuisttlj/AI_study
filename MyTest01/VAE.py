import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

class Encoder_Net:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=[784, 128], stddev=tf.sqrt(1 / 784)))
        self.b1 = tf.Variable(tf.zeros(128))
        self.w2_mu = tf.Variable(tf.truncated_normal(shape=[128, 100], stddev=tf.sqrt(2 / 128)))
        self.b2_mu = tf.Variable(tf.zeros(100))
        self.w2_sigma = tf.Variable(tf.truncated_normal(shape=[128, 100], stddev=tf.sqrt(2 / 128)))
        self.b2_sigma = tf.Variable(tf.zeros(100))

    def forward(self, x):
        a1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        z2_mu = tf.matmul(a1, self.w2_mu) + self.b2_mu
        z2_logvar = tf.matmul(a1, self.w2_sigma) + self.b2_sigma
        return z2_mu, z2_logvar


class Decoder_Net:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=[100, 128], stddev=tf.sqrt(1 / 100)))
        self.b1 = tf.Variable(tf.zeros(128))
        self.w2 = tf.Variable(tf.truncated_normal(shape=[128, 784], stddev=tf.sqrt(2 / 128)))
        self.b2 = tf.Variable(tf.zeros(784))

    def forward(self, x):
        a1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        output = tf.matmul(a1, self.w2) + self.b2
        real_out = tf.nn.sigmoid(output)
        return output, real_out


class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

        self.encoder = Encoder_Net()
        self.decoder = Decoder_Net()

    def forward(self):
        # self.samples = self.decoder.forward(self.z)
        self.mu, self.logvar = self.encoder.forward(self.x)
        sample = tf.random_normal(shape=[64, 100])
        self.var = tf.exp(self.logvar)
        std = tf.sqrt(self.var)
        _x = self.mu + std * sample
        self.output, _ = self.decoder.forward(_x)

    def decode(self):
        I = tf.random_normal(shape=[16, 100])
        _, real_out = self.decoder.forward(I)
        return real_out

    def backward(self):
        loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.output), axis=1)
        # loss1 = tf.reduce_mean((self.output - self.x) ** 2)
        loss2 = 0.5 * tf.reduce_sum(self.var + self.mu ** 2 - 1 - self.logvar, axis=1)
        self.loss = tf.reduce_mean(loss1 + loss2)

        # loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.output))
        # loss2 = tf.reduce_mean(0.5 * (self.var + self.mu ** 2 - 1 - self.logvar))
        # self.loss = loss1 + loss2

        self.op = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()

    if not os.path.exists('out1/'):
        os.makedirs('out1/')

    test_output = net.decode()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # plt.ion()
        for i in range(1000000):
            xs, _ = mnist.train.next_batch(64)
            _loss, _ = sess.run([net.loss, net.op], feed_dict={net.x: xs})

            if i % 1000 == 0:
                print(_loss)
                test_img_data = sess.run(test_output)
                # test_img = np.reshape(np.array(test_img_data), [28, 28])
        #         plt.imshow(test_img,  cmap='Greys_r')
        #         plt.pause(0.1)
        # plt.ioff()
                fig = plot(test_img_data)
                plt.savefig('out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)
