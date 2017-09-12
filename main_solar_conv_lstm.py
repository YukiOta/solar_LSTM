# coding: utf-8
"""CONV_LSTMを実装する
input: image
output: 画像生成か？？
"""

import os.path
import time
import numpy as np
import tensorflow as tf
import cv2
 
import layer_def as ldef
import BasicConvLSTMCell

# パラメータの設定
FLAGS = tf.app.flags.FLAGS  # 最初のおまじない
tf.app.flags.DEFINE_string(
    'train_dir',
    './checkpoints/train_store_conv_lstm',
    """dir to store trained net"""
    )
tf.app.flags.DEFINE_integer(
    'seq_length',
    10,
    """size of hidden layer"""
)
tf.app.flags.DEFINE_integer(
    'seq_start',
    5,
    """start of seq generation"""
)
tf.app.flags.DEFINE_integer(
    'max_step',
    200000,
    """max number of steps"""
)
tf.app.flags.DEFINE_float(
    'keep_prob',
    .8,
    """for Dropout"""
)
tf.app.flags.DEFINE_float(
    'lr',
    .001,
    """learning rate"""
)
tf.app.flags.DEFINE_integer(
    'batch_size',
    16,
    """batch size of training"""
)
tf.app.flags.DEFINE_float(
    'weight_init',
    .1,
    """weight initilaizaion for fully connected layers"""
)

# ビデオを生成するためっぽい
# 動くか不安ではある
fourcc = cv2.cv.CV_FOURCC("m", "p", "4", "v")

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
    dat = np.zeros((batch_size, seq_length, shape, shape, 3))






























#end
