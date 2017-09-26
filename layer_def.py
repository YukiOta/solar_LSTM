# coding: utf-8
"""CONV_LSTMにおけるレイヤーの定義

"""

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('weight_decay', 0.0005, """ """)

a = np.zeros((100, 100, 3))
matrix1 = tf.constant(a)
matrix1.get_shape()[2]

def conv/layer(inputs, kernel_size, stride, num_features, idx, linear=False):
    with tf.variable_scope('{0}_conv'.format(idx)) as scope:
        input_channels = inputs.get_shape()[3]

        weights = 


















# end
