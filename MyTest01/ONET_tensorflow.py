import tensorflow as tf
from sampling_tensorflow import FaceDataset
import os


class ONet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 48, 48, 3))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 5))
        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=tf.sqrt(1 / 16)))
        self.conv1_b = tf.Variable(tf.zeros(32))
        self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=tf.sqrt(1 / 32)))
        self.conv2_b = tf.Variable(tf.zeros(64))
        self.conv3_w = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=tf.sqrt(1 / 32)))
        self.conv3_b = tf.Variable(tf.zeros(64))
        self.conv4_w = tf.Variable(tf.truncated_normal([2, 2, 64, 128], stddev=tf.sqrt(1 / 64)))
        self.conv4_b = tf.Variable(tf.zeros(128))
        self.fc5_w = tf.Variable(tf.truncated_normal([128 * 2 * 2, 256], stddev=tf.sqrt(1 / 128)))
        self.fc5_b = tf.Variable(tf.zeros(256))
        self.fc6_1_w = tf.Variable(tf.truncated_normal([256, 1], stddev=tf.sqrt(2 / 1)))
        self.fc6_1_b = tf.Variable(tf.zeros(1))
        self.fc6_2_w = tf.Variable(tf.truncated_normal([256, 4], stddev=tf.sqrt(1 / 2)))
        self.fc6_2_b = tf.Variable(tf.zeros(4))

    def forwardprop(self):
        self.conv1 = tf.nn.conv2d(self.x, self.conv1_w, [1, 1, 1, 1], padding="VALID") + self.conv1_b
        self.batch_norm1 = tf.layers.batch_normalization(self.conv1)
        self.relu_conv1 = tf.nn.leaky_relu(self.batch_norm1)
        self.pool1 = tf.nn.max_pool(self.relu_conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

        self.conv2 = tf.nn.conv2d(self.pool1, self.conv2_w, [1, 1, 1, 1], padding="VALID") + self.conv2_b
        self.batch_norm2 = tf.layers.batch_normalization(self.conv2)
        self.relu_conv2 = tf.nn.leaky_relu(self.batch_norm2)
        self.pool2 = tf.nn.max_pool(self.relu_conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

        self.conv3 = tf.nn.conv2d(self.pool2, self.conv3_w, [1, 1, 1, 1], padding="VALID") + self.conv3_b
        self.batch_norm3 = tf.layers.batch_normalization(self.conv3)
        self.relu_conv3 = tf.nn.leaky_relu(self.batch_norm3)
        self.pool3 = tf.nn.max_pool(self.relu_conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        self.conv4 = tf.nn.conv2d(self.pool3, self.conv4_w, [1, 1, 1, 1], padding="VALID") + self.conv4_b
        self.batch_norm4 = tf.layers.batch_normalization(self.conv4)
        self.relu_conv4 = tf.nn.leaky_relu(self.batch_norm4)

        self.flat = tf.reshape(self.relu_conv4, [-1, 128 * 2 * 2])

        self.fc5 = tf.matmul(self.flat, self.fc5_w) + self.fc5_b
        self.batch_norm5 = tf.layers.batch_normalization(self.fc5)
        self.relu_fc5 = tf.nn.leaky_relu(self.batch_norm5)

        self.fc6_1 = tf.matmul(self.relu_fc5, self.fc6_1_w) + self.fc6_1_b
        self.output_cond = self.fc6_1

        self.fc6_2 = tf.matmul(self.relu_fc5, self.fc6_2_w) + self.fc6_2_b
        self.output_offset = self.fc6_2

        self.cond_mask = tf.where(self.y[:, 0] < 2)
        self.loss_cond = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.gather_nd(self.output_cond[:, 0], self.cond_mask),
                                                    labels=tf.gather_nd(self.y[:, 0], self.cond_mask)))

        self.offset_mask = tf.where(self.y[:, 0] > 0)
        self.loss_offset = tf.losses.mean_squared_error(tf.gather_nd(self.output_offset, self.offset_mask),
                                                        tf.gather_nd(self.y[:, 1:], self.offset_mask))

        self.loss = self.loss_cond + self.loss_offset

    def backprop(self):
        self.bp = tf.train.AdamOptimizer(0.001).minimize(self.loss)


if __name__ == '__main__':
    o_net = ONet()
    o_net.forwardprop()
    o_net.backprop()
    init = tf.global_variables_initializer()
    face_data = FaceDataset(path=r"E:\myceleba\48", batch_num=512, epoch=15)
    if not os.path.exists(r"./param_o_tensorflow"):
        os.makedirs(r"./param_o_tensorflow")
    save_path = r"./param_o_tensorflow/rnet.ckpt"

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if os.path.exists(save_path + ".data-00000-of-00001"):
            saver.restore(sess, save_path)
        else:
            sess.run(init)

        for i in range(100000):
            xs, ys = face_data.get_batch(sess)
            _loss_cond, _loss_offset, _loss, _ = sess.run([o_net.loss_cond, o_net.loss_offset, o_net.loss, o_net.bp],
                                                          feed_dict={o_net.x: xs, o_net.y: ys})
            print("loss", _loss, " cond_loss:", _loss_cond, " offset_loss:", _loss_offset)
            saver.save(sess, save_path)
            print("save success")
