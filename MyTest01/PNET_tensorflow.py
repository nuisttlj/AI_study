import tensorflow as tf
from sampling_tensorflow import FaceDataset
import os


class PNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 12, 12, 3))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 5))
        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 3, 10], stddev=tf.sqrt(1 / 5)))
        self.conv1_b = tf.Variable(tf.zeros(10))
        self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 10, 16], stddev=tf.sqrt(1 / 8)))
        self.conv2_b = tf.Variable(tf.zeros(16))
        self.conv3_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=tf.sqrt(1 / 16)))
        self.conv3_b = tf.Variable(tf.zeros(32))
        self.conv4_1_w = tf.Variable(tf.truncated_normal([1, 1, 32, 1], stddev=tf.sqrt(2 / 1)))
        self.conv4_1_b = tf.Variable(tf.zeros(1))
        self.conv4_2_w = tf.Variable(tf.truncated_normal([1, 1, 32, 4], stddev=tf.sqrt(1 / 2)))
        self.conv4_2_b = tf.Variable(tf.zeros(4))

    def forwardprop(self):
        self.conv1 = tf.nn.conv2d(self.x, self.conv1_w, [1, 1, 1, 1], padding="VALID") + self.conv1_b
        self.batch_norm1 = tf.layers.batch_normalization(self.conv1)
        self.relu_conv1 = tf.nn.leaky_relu(self.batch_norm1)
        self.pool1 = tf.nn.max_pool(self.relu_conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        self.conv2 = tf.nn.conv2d(self.pool1, self.conv2_w, [1, 1, 1, 1], padding="VALID") + self.conv2_b
        self.batch_norm2 = tf.layers.batch_normalization(self.conv2)
        self.relu_conv2 = tf.nn.leaky_relu(self.batch_norm2)

        self.conv3 = tf.nn.conv2d(self.relu_conv2, self.conv3_w, [1, 1, 1, 1], padding="VALID") + self.conv3_b
        self.batch_norm3 = tf.layers.batch_normalization(self.conv3)
        self.relu_conv3 = tf.nn.leaky_relu(self.batch_norm3)

        self.conv4_1 = tf.nn.conv2d(self.relu_conv3, self.conv4_1_w, [1, 1, 1, 1], padding="VALID") + self.conv4_1_b
        self.output_cond = tf.reshape(self.conv4_1, [-1, 1])

        self.conv4_2 = tf.nn.conv2d(self.relu_conv3, self.conv4_2_w, [1, 1, 1, 1], padding="VALID") + self.conv4_2_b
        self.output_offset = tf.reshape(self.conv4_2, [-1, 4])

        self.cond_mask = tf.where(self.y[:, 0] < 2)
        self.loss_cond = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.gather_nd(self.output_cond[:, 0], self.cond_mask),
                                                    labels=tf.gather_nd(self.y[:, 0], self.cond_mask)))

        self.offset_mask = tf.where(self.y[:, 0] > 0)
        self.loss_offset = tf.reduce_mean(
            (tf.gather_nd(self.output_offset, self.offset_mask) - tf.gather_nd(self.y[:, 1:], self.offset_mask)) ** 2)

        self.loss = self.loss_cond + self.loss_offset

    def backprop(self):
        self.bp = tf.train.AdamOptimizer(0.01).minimize(self.loss)


if __name__ == '__main__':
    p_net = PNet()
    p_net.forwardprop()
    p_net.backprop()
    init = tf.global_variables_initializer()
    face_data = FaceDataset(path=r"E:\myceleba\12", batch_num=512, epoch=15)
    if not os.path.exists(r"./param_p_tensorflow"):
        os.makedirs(r"./param_p_tensorflow")
    save_path = r"./param_p_tensorflow/pnet.ckpt"

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if os.path.exists(save_path+".data-00000-of-00001"):
            saver.restore(sess, save_path)
        else:
            sess.run(init)

        for i in range(100000):
            xs, ys = face_data.get_batch(sess)
            _loss_cond, _loss_offset, _loss, _ = sess.run([p_net.loss_cond, p_net.loss_offset, p_net.loss, p_net.bp],
                                                          feed_dict={p_net.x: xs, p_net.y: ys})
            print("loss", _loss, " cond_loss:", _loss_cond, " offset_loss:", _loss_offset)
            saver.save(sess, save_path)
            print("save success")
