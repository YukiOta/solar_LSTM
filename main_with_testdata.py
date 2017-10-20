# coding: utf-8

import time
import os
import datetime as dt
import h5py
import argparse
import Load_data as ld
import main_LSTM as LSTM
import numpy as np


def main(save):
    model, img_train, img_gtruth = LSTM.train_convLSTM_with_test(save)
    img_test = img_train[1004]
    LSTM.predict_convLSTM2D(model, img_test, save, date="test_data", start=0)
    print("done")

if __name__ == '__main__':
    SAVE_dir = "./RESULT/LSTM_test_2017_1020/"
    if not os.path.isdir(SAVE_dir):
        os.makedirs(SAVE_dir)
    # 時間の表示
    start = time.time()
    main(SAVE_dir)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time)+" [sec]")
