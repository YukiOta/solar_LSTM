# coding: utf-8

import time
import os
import datetime as dt
import h5py
import argparse
import Load_data as ld
import main_LSTM as LSTM
import numpy as np


def main(img, target, save):
    '''
    ここで、実験設定について書くテキストファイルを書きだす
    '''
    im_size = 100
    with open(save + "setting.txt", "w") as f:
        f.write("SAVE DIR: " + save + "\n")
        f.write("IMG  DIR: " + img + "\n")
        f.write("TARGET DIR: " + target + "\n")
        f.write("Im Size: "+str(im_size))
        f.write("\n")

    img_tr, target_tr, date_list = ld.Load_data(
        img, target, save, im_size=im_size)
    for i, day in enumerate(date_list):
        if day == "20170518":
            print(day + " is taken")
            print("index is ", i)
            index_test = i
    img_test = img_tr.pop(index_test)
    date_test = date_list.pop(index_test)
    model = LSTM.training_convLSTM2D(img_tr, target_tr, date_test, save)
    LSTM.predict_convLSTM2D(model, img_test, save, date_test)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="../data/PV_IMAGE/",
        help="choose your data (image) directory"
    )
    parser.add_argument(
        "--target_dir",
        default="../data/PV_CSV/",
        help="choose your target dir"
    )
    args = parser.parse_args()
    DATA_DIR, TARGET_DIR = args.data_dir, args.target_dir
    SAVE_dir = "./RESULT/AIST_LSTM/" + dt.datetime.today().strftime("%Y_%m_%d") + "/"
    if not os.path.isdir(SAVE_dir):
        os.makedirs(SAVE_dir)
    # 時間の表示
    start = time.time()
    main(DATA_DIR, TARGET_DIR, SAVE_dir)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + " [sec]")
