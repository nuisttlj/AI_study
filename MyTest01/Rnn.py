import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class Rnn:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.w1 = tf.Variable(tf.truncated_normal(shape=[28, 28], dtype=tf.float32, stddev=tf.sqrt(1 / 28)))
        self.b1 = tf.Variable(tf.zeros([28]))
        self.out_w = tf.Variable(tf.truncated_normal(shape=[28, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 28)))
        self.out_b = tf.Variable(tf.zeros([10]))

    def forward(self):
        x = tf.reshape(self.x, (-1, 28))
        out1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        out1 = tf.reshape(out1, (-1, 28, 28))
        cell = tf.nn.rnn_cell.BasicLSTMCell(28)
        init_state = cell.zero_state(100, dtype=tf.float32)
        output, final_state = tf.nn.dynamic_rnn(cell, out1, initial_state=init_state)
        output = output[:, -1, :]
        self.final_output = tf.nn.softmax(tf.matmul(output, self.out_w) + self.out_b)

    def backward(self):
        self.loss = tf.reduce_mean((self.final_output - self.y) ** 2)
        self.op = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    net = Rnn()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000000):
            xs, ys = mnist.train.next_batch(100)
            _loss, _ = sess.run([net.loss, net.op], feed_dict={net.x: xs, net.y: ys})
            if i % 10 == 0:
                print(_loss)
            if i % 100 == 0:
                xss, yss = mnist.validation.next_batch(100)
                _loss_, final_output = sess.run([net.loss, net.final_output], feed_dict={net.x: xss, net.y: yss})
                accuracy = np.mean(np.array(np.argmax(final_output, axis=1) == np.argmax(yss, axis=1), dtype=np.float32))
                print(accuracy)

