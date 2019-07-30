import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datasets

batch_size = 64


class EncoderNet:
    # def __init__(self):
    #     self.w1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[60 * 3, 256], stddev=tf.sqrt(1 / (60 * 3))))
    #     self.b1 = tf.Variable(tf.zeros(256))
    #
    # def forward(self, x):
    #     with tf.variable_scope("Encodernet") as scope:
    #         x = tf.transpose(x, (0, 2, 1, 3))
    #         x = tf.reshape(x, [-1, 60 * 3])
    #         y1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
    #         y1 = tf.reshape(y1, [-1, 120, 256])
    #         cell = tf.nn.rnn_cell.LSTMCell(256)
    #         init_state = cell.zero_state(batch_size, dtype=tf.float32)
    #         encoder_out, encoder_finalstate = tf.nn.dynamic_rnn(cell, y1, initial_state=init_state)
    #         encoder_out = encoder_out[:, -1, :]
    #         return encoder_out
    #         # return encoder_finalstate
    def forward(self, x):
        with slim.arg_scope([slim.conv2d], padding="SAME", activation_fn=tf.nn.relu):
            net = slim.stack(x, slim.conv2d, [(16, 3, 1), (16, 3, 2), (32, 3), (32, 3, 2), (16, 3), (16, 3, 2)])
            # net = slim.conv2d(x, 16, 3,
            #                   weights_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / (3 * 3 * 3))),
            #                   scope="conv1")
            # net = slim.conv2d(net, 16, 3, stride=2,
            #                   weights_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(2 / (3 * 3 * 16))),
            #                   scope="conv2")
            # net = slim.conv2d(net, 32, 3,
            #                   weights_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(2 / (3 * 3 * 16))),
            #                   scope="conv3")
            # net = slim.conv2d(net, 32, 3, stride=2,
            #                   weights_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(2 / (3 * 3 * 32))),
            #                   scope="conv4")
            # net = slim.conv2d(net, 16, 3,
            #                   weights_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(2 / (3 * 3 * 32))),
            #                   scope="conv5")
            # net = slim.conv2d(net, 16, 3, stride=2,
            #                   weights_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(2 / (3 * 3 * 16))),
            #                   scope="conv6")
        return net


class DecoderNet:
    def __init__(self):
        self.w_out = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[256, 62], stddev=tf.sqrt(2 / 256)))
        self.b_out = tf.Variable(tf.zeros(62))

    def forward(self, y):
        with tf.variable_scope("Decodernet") as scope:
            y = tf.reshape(y, [batch_size, 1, 15*5*16])
            y = tf.tile(y, [1, 6, 1])
            # input_x = tf.zeros(shape=[batch_size, 4, 256], dtype=tf.float32)
            cell = tf.nn.rnn_cell.LSTMCell(256)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            decoder_out, decoder_finalstate = tf.nn.dynamic_rnn(cell, y, initial_state=init_state)
            # decoder_out, decoder_finalstate = tf.nn.dynamic_rnn(cell, input_x, initial_state=y)
            decoder_out = tf.reshape(decoder_out, [-1, 256])
            decoder_out = tf.matmul(decoder_out, self.w_out) + self.b_out
            decoder_out = tf.reshape(decoder_out, [-1, 6, 62])
            return decoder_out


class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 40, 120, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 6, 62])

    def forward(self):
        encodernet = EncoderNet()
        charcter = encodernet.forward(self.x)
        decodernet = DecoderNet()
        self.output = decodernet.forward(charcter)

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output))
        self.op = tf.train.AdamOptimizer().minimize(self.loss)

    def accuracy(self):
        bool1 = tf.equal(tf.argmax(self.output, axis=2), tf.argmax(self.y, axis=2))
        accuracy = tf.reduce_mean(tf.map_fn(lambda total_num: tf.cast(tf.equal(total_num, 6), dtype=tf.float32),
                                            tf.reduce_sum(tf.cast(bool1, dtype=tf.float32), axis=1)))
        return accuracy

    def compute_out(self, x):
        final_out = ""
        for j in range(6):
            index = np.argmax(x[0][j])
            if index <= 9:
                final_out += str(index)
            elif index <= 35:
                final_out += chr(index + 55)
            else:
                final_out += chr(index + 61)
        return final_out


if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    net.accuracy()
    dataset = datasets.MyDataset(r"C:\vertification", batch_size)
    dataset_test = datasets.MyDataset(r"C:\vertification_test", batch_size)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(10000):
            xs, ys = dataset.get_batch(sess)
            _loss, _ = sess.run([net.loss, net.op], feed_dict={net.x: xs, net.y: ys})
            if i % 10 == 0:
                print(_loss)
                xss, yss = dataset_test.get_batch(sess)
                _output, _, _accuracy = sess.run([net.output, net.op, net.accuracy()],
                                                 feed_dict={net.x: xss, net.y: yss})
                print(_accuracy)
                print(net.compute_out(_output))
                print(net.compute_out(yss))
                if _accuracy == 1:
                    print(i)
                    exit()
