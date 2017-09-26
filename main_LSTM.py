# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import Load_data as ld
import h5py
# import seaborn as sns

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Convolution3D, ZeroPadding3D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization


def CNN_model3(
        activation="relu",
        loss="mean_squared_error",
        optimizer="Adadelta",
        layer=0,
        height=0,
        width=0):
    """
    INPUT -> [CONV -> RELU] -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(
        layer, height, width), data_format="channels_first"))
    model.add(Convolution2D(filters=16, kernel_size=3, border_mode='same',
                            data_format="channels_first"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def data_plot(model, target, img, SAVE_dir, batch_size=10, date="hoge", save_csv=True):

    num = []
    time = []
    for i in range(target[:, 0].shape[0]):
        if i % 50 == 0:
            num.append(i)
            time.append(target[:, 0][i])
        if i == target[:, 0].shape[0] - 1:
            num.append(i)
            time.append(target[:, 0][i])
    img_ = img.transpose(0, 3, 1, 2).copy()
    pred = model.predict(img_, batch_size=batch_size, verbose=1).copy()
    # if pred.shape:
    #     print(pred.shape)
    # if type(pred):
    #     print(type(pred))
    # print(target[1].shape)
    plt.figure()
    plt.plot(pred, label="Predicted")
    plt.plot(target[:, 1], label="Observed")
    plt.legend(loc='best')
    plt.title("Prediction tested on" + date)
    plt.xlabel("Time")
    plt.ylabel("Generated Power[kW]")
    plt.ylim(0, 25)
    plt.xticks(num, time)

    pred_ = pred.reshape(pred.shape[0])
    if save_csv is True:
        save_target_and_prediction(
            target=target[:, 1], pred=pred_, title=date, SAVE_dir=SAVE_dir)

    filename = date + "_data"
    i = 0
    while os.path.exists(SAVE_dir + '{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig(SAVE_dir + '{}{:d}.png'.format(filename, i))


def save_target_and_prediction(target, pred, title, SAVE_dir):

    pred_df = pd.DataFrame(pred)
    target_df = pd.DataFrame(target)
    df = pd.concat([target_df, pred_df], axis=1)
    df.columns = ["TARGET", "PREDICTION"]
    df.to_csv(SAVE_dir + "Error_csv_" + title + ".csv")


def training_conv2D(img_tr, target_tr, date_list, SAVE_dir):

    # for i in range(len(date_list)):
    for i in range(1):

        print("-----Training on " + str(date_list[i]) + "-----")
        training_start_time = time.time()

        ts_img = 0
        ts_target = 0
        ts_img_pool = 0
        ts_target_pool = 0
        # i=1
        ts_img_pool = img_tr.pop(i)
        ts_target_pool = target_tr.pop(i)
        ts_img = ts_img_pool.copy()
        ts_target = ts_target_pool.copy()

        img_tr_all = 0
        target_tr_all = 0
        img_tr_all = np.concatenate((
            img_tr[:]
        ), axis=0)
        target_tr_all = np.concatenate((
            target_tr[:]
        ), axis=0)

        # テストデータと訓練データから、訓練データでとった平均を引く
        mean_img = ld.compute_mean(image_array=img_tr_all)
        np.save(SAVE_dir + "mean_array_" +
                str(date_list[i]) + ".npy", mean_img)
        img_tr_all -= mean_img
        ts_img -= mean_img

        # transpose for CNN INPUT shit
        img_tr_all = img_tr_all.transpose(0, 3, 1, 2)
        print(img_tr_all.shape)
        # set image size
        layer = img_tr_all.shape[1]
        height = img_tr_all.shape[2]
        width = img_tr_all.shape[3]

        print("Image and Target Ready")

        # parameter
        activation = ["relu", "sigmoid"]
        optimizer = ["adam", "adadelta", "rmsprop"]
        nb_epoch = [10, 25, 50]
        batch_size = [5, 10, 15]

        # model set
        model = None
        model = CNN_model3(
            activation="relu",
            optimizer="Adadelta",
            layer=layer,
            height=height,
            width=width)
        # plot_model(model, to_file='CNN_model.png')

        # initialize check
        data_plot(
            model=model, target=ts_target, img=ts_img, SAVE_dir=SAVE_dir, batch_size=10,
            date=date_list[i], save_csv=True)

        early_stopping = EarlyStopping(patience=3, verbose=1)

        # Learning model
        hist = model.fit(img_tr_all, target_tr_all[:, 1],
                         epochs=nb_epoch[0],
                         batch_size=batch_size[1],
                         validation_split=0.1,
                         callbacks=[early_stopping])
        data_plot(
            model=model, target=ts_target, img=ts_img, SAVE_dir=SAVE_dir, batch_size=10,
            date=date_list[i], save_csv=True)
        # evaluate
        try:
            img_tmp = ts_img.transpose(0, 3, 1, 2)
            score = model.evaluate(img_tmp, ts_target[:, 1], verbose=1)
            print("Evaluation " + date_list[i])
            print('TEST LOSS: ', score[0])
            test_error_list.append(score[0])
        except:
            print("error in evaluation")

        try:
            model.save(SAVE_dir + "model_{}".format(str(date_list[i])) + ".h5")
        except:
            print("error in save model")

        # put back data
        img_tr.insert(i, ts_img_pool)
        target_tr.insert(i, ts_target_pool)

        tr_elapsed_time = time.time() - training_start_time
        print("elapsed_time:{0}".format(tr_elapsed_time) + " [sec]")

    # error_lossの日を保存
   #  with open(SAVE_dir + "test_loss.txt", "w") as f:
   #      f.write(str(test_error_list))
    return model


def training_convLSTM2D(img_tr, target_tr, date_list, SAVE_dir):
    """
    kerasでConvLSTMを実装する。
    input : img_tr, target_tr, date_list, SAVE_dir
    out : img?
    """
    def CNN_convLSTM(
            activation="relu",
            loss="binary_crossentropy",
            optimizer="Adadelta",
            layer=0,
            height=0,
            width=0):
        """
        INPUT -> [CONV -> RELU] -> OUT
        """
        model = Sequential()

        model.add(ZeroPadding3D((1, 1, 1), data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=16, kernel_size=3, padding='same',
                             activation=activation, data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(Convolution3D(filters=1, kernel_size=2, ))

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model
