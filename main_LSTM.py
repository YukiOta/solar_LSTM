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
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, Convolution3D
from keras.layers.core import Dense, Activation, Flatten
from keras.callbacks import EarlyStopping, TensorBoard
# from keras.optimizers import Adam, Adadelta, RMSprop
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


def array2LSTM(array_1, num_frame=15):
    '''
    input
    -----------
    array_1 (timesteps, 100, 100, 3)
    output
    -----------
    array_2 (batch, num_frames, 100, 100, 3)
    '''
    count = 0
    if array_1.ndim == 4:
        print("add new axis")
        array_1 = array_1[np.newaxis, :]
    timesteps = array_1.shape[1]
    for i in range(timesteps):
        if i <= timesteps - num_frame:
            tmp = array_1[:, i:i + num_frame, :, :, :]
            if count == 0:
                array_2 = tmp
                count += 1
            else:
                array_2 = np.concatenate((array_2, tmp), axis=0)
        else:
            break
    return array_2


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
            # test_error_list.append(score[0])
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
    img_tr : np.array (days, timesteps, height, width, layer)
    target_tr :
    date_list : daysのリスト
    SAVE_dir : 結果を保存するディレクトリ
    ------------------------
    out : img?
    """
    def CNN_convLSTM(
            activation="relu",
            loss="binary_crossentropy",
            optimizer="Adadelta",
            layer=0,
            height=0,
            width=0,
            days=0,
            timesteps=0):
        """
        INPUT -> [CONV -> RELU] -> OUT
        """
        model = Sequential()

        # model.add(ZeroPadding3D((1, 1, 1), data_format="channels_last", input_shape=(timesteps, height, width, layer)))
        # model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same',
                             activation=activation, data_format="channels_last",
                             input_shape=(None, height, width, layer),
                             return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same',
                             activation=activation, data_format="channels_last",
                             return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same',
                             activation=activation, data_format="channels_last",
                             return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same',
                             activation=activation, data_format="channels_last",
                             return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same',
                             activation=activation, data_format="channels_last",
                             return_sequences=True))
        model.add(BatchNormalization())
        model.add(Convolution3D(filters=3, kernel_size=(3, 3, 3),
                                padding="same", data_format="channels_last"))

        model.compile(loss=loss, optimizer=optimizer)
        return model

    """
    データの読み込みとground truthの作成
    画像は、(day, timstep, 100, 100, 3)で受け取ってる
    """
    num_frame = 15
    for index in range(len(img_tr)):
        img_tra = img_tr[index][:-1, :, :, :]
        img_gt = img_tr[index][1:, :, :, :]
        img_tmp_tr = array2LSTM(img_tra, num_frame)
        img_tmp_gt = array2LSTM(img_gt, num_frame)
        if index == 0:
            img_train = img_tmp_tr
            img_gtruth = img_tmp_gt
        else:
            img_train = np.concatenate((img_train, img_tmp_tr), axis=0)
            img_gtruth = np.concatenate((img_gtruth, img_tmp_gt), axis=0)

    """
    データのトレーニング
    """
    days = img_train.shape[0]
    timesteps = img_train.shape[1]
    h = img_train.shape[2]
    w = img_train.shape[3]
    l = img_train.shape[4]

    batch_size = 10
    epoch = 1000
    validation_split = 0.2

    model = CNN_convLSTM(
        activation="relu",
        loss="binary_crossentropy",
        optimizer="Adadelta",
        height=h, width=w, layer=l, days=days, timesteps=timesteps
    )
    with open(SAVE_dir + "setting.txt", "a") as f:
        f.write("----about experiment setting-----\n")
        f.write("(day, timesteps, h, w, l) : " +
                str((days, timesteps, h, w, l)) + "\n")
        f.write("batch size: " + str(batch_size) + "\n")
        f.write("epoch: " + str(epoch) + "\n")
        f.write("validation_split: " + str(validation_split) + "\n")
    hist = model.fit(
        img_train, img_gtruth, batch_size=batch_size, epochs=epoch, validation_split=validation_split, verbose=1, callbacks=[TensorBoard(log_dir='./log/solar')])
    try:
        model.save(SAVE_dir + "model_{}".format(str(date_list)) + ".h5")
    except:
        print("ahhhhhhh error in model.save")
    return model


def train_convLSTM_with_test(SAVE_dir):
    """
    kerasでConvLSTMを実装する。
    input : SAVE_dir
    SAVE_dir : 結果を保存するディレクトリ
    ------------------------
    out : img?
    """
    def CNN_convLSTM(
            activation="relu",
            loss="binary_crossentropy",
            optimizer="Adadelta",
            layer=0,
            height=0,
            width=0,
            days=0,
            timesteps=0):
        """
        INPUT -> [CONV -> RELU] -> OUT
        """
        model = Sequential()

        # model.add(ZeroPadding3D((1, 1, 1), data_format="channels_last", input_shape=(timesteps, height, width, layer)))
        # model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same',
                             activation=activation, data_format="channels_last",
                             input_shape=(None, height, width, layer),
                             return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same',
                             activation=activation, data_format="channels_last",
                             return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same',
                             activation=activation, data_format="channels_last",
                             return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same',
                             activation=activation, data_format="channels_last",
                             return_sequences=True))
        model.add(BatchNormalization())
        model.add(Convolution3D(filters=layer, kernel_size=(3, 3, layer),
                                padding="same", data_format="channels_last"))

        model.compile(loss=loss, optimizer=optimizer)
        return model

    def make_date():
        '''
        人工データの作成。
        '''
        # test
        time = 20
        row = 80
        col = 80
        filters = 1
        training = 5000
        train = np.zeros((training, time, row, col, 1), dtype=np.float)
        gt = np.zeros((training, time, row, col, 1), dtype=np.float)
        # for i in range(1000):
        #    gt[::,0,0,0] = np.random.random()

        for i in range(training):
            n = np.random.randint(3, 8)
            # n=15
            for j in range(n):
                xstart = np.random.randint(20, 60)
                ystart = np.random.randint(20, 60)
                directionx = np.random.randint(0, 3) - 1
                directiony = np.random.randint(0, 3) - 1
                directionx = np.random.randint(0, 3) - 1
                gravity = 0  # np.random.randint(0,3) - 1
                w = np.random.randint(2, 4)
                # rint directionx,directiony
                for t in range(time):
                    # w = 2
                    train[i, t, xstart + directionx * t - w:xstart + directionx * t + w,
                          ystart + directiony * t + int(0.1 * gravity * t**2) - w:ystart + directiony * t + int(0.1 * gravity * t**2) + w, 0] += 1

                    # Make it more robust
                    # Noise
                    if np.random.randint(0, 2):
                        train[i, t, xstart + directionx * t - w - 1:xstart + directionx * t + w + 1,
                              ystart + directiony * t + int(0.1 * gravity * t**2) - w - 1:ystart + directiony * t + int(0.1 * gravity * t**2) + w + 1, 0] += 0.1

                    if np.random.randint(0, 2):
                        train[i, t, xstart + directionx * t - w + 1:xstart + directionx * t + w - 1,
                              ystart + directiony * t + int(0.1 * gravity * t**2) + w - 1:ystart + directiony * t + int(0.1 * gravity * t**2) + w - 1, 0] -= 0.1

                    gt[i, t, xstart + directionx * (t + 1) - w:xstart + directionx * (t + 1) + w,
                       ystart + directiony * (t + 1) + int(0.1 * gravity * (t + 1)**2) - w:ystart + directiony * (t + 1) + int(0.1 * gravity * (t + 1)**2) + w, 0] += 1

        train = train[::, ::, 20:60, 20:60, ::]
        gt = gt[::, ::, 20:60, 20:60, ::]
        train[train >= 1] = 1
        gt[gt >= 1] = 1
        return train, gt

    """
    データのトレーニング
    """
    img_train, img_gtruth = make_date()

    days = img_train.shape[0]
    timesteps = img_train.shape[1]
    h = img_train.shape[2]
    w = img_train.shape[3]
    l = img_train.shape[4]

    model = CNN_convLSTM(
        activation="relu",
        loss="binary_crossentropy",
        optimizer="Adadelta",
        height=h, width=w, layer=l, days=days, timesteps=timesteps
    )
    hist = model.fit(
        img_train[:1000:], img_gtruth[:1000], batch_size=10, epochs=1000, validation_split=0.05, verbose=0, callbacks=[TensorBoard(log_dir='./log/test')])
    try:
        model.save(SAVE_dir + "model_test_set.h5")
    except:
        print("ahhhhhhh error in model.save")
    return model, img_train, img_gtruth


def predict_convLSTM2D(model, img_test, SAVE_DIR, date, start=30):
    """
    画像の予測を行う
    input:
    model : 学習したモデル
    img_test : テストデータ
    target_ts : テストターゲットデータ

    output:
    imageをplt.show
    """
    start = start
    frame = 10
    print("start predicting from %d to %d" % (start, start + frame))
    track = img_test[start:start + frame, :, :, :]
    for i in range(16):
        new_pos = model.predict(track[np.newaxis, :, :, :, :])
        print(track.shape, new_pos[0, -1, :, :, :].shape)
        new = new_pos[:, -1, :, :, :]
        new[new > 0.5] = 1
        new[new <= 0.5] = 0
        track = np.concatenate((track, new), axis=0)

    for i in range(15):
        gt_im = img_test[start + i, :, :, 0]
        test_im = track[i, :, :, 0]

        plt.clf()
        fig = plt.figure(figsize=(10, 5))

        """
        予測画像の表示
        """
        ax = fig.add_subplot(121)
        ax.imshow(test_im, cmap="gray", interpolation="none")

        if i >= frame:
            ax.text(37, 3, "Prediction", color="orange", fontdict={
                    "fontsize": 15, "fontweight": 'bold', "ha": "right", "va": "center"})
        else:
            ax.text(37, 3, "Initial Time", color="white", fontdict={
                    "fontsize": 15, "fontweight": 'bold', "ha": "right", "va": "center"})
        plt.xticks([])
        plt.yticks([])

        """
        Ground Truthの表示
        """
        ax = fig.add_subplot(122)
        ax.imshow(gt_im, cmap="gray", interpolation="none")
        ax.text(37, 3, "Ground Truth", color="white", fontdict={
                "fontsize": 15, "fontweight": 'bold', "ha": "right", "va": "center"})
        plt.xticks([])
        plt.yticks([])

        plt.savefig(SAVE_DIR + "animate_" + date + "_%i" %
                    (i + 1), cmap="gray", interpolation="none")
        plt.clf()
