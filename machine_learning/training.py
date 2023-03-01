"""
Trains a DNN with a numpy array with variable data columns to distinguish between pions and Kaons given multiple variables (features) on which to train simultaneously
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Normalization, AlphaDropout
from keras.models import Model
from keras.optimizers import Adam
from machine_learning.dnn_utils import dnn_settings
from data.import_functions import get_txtpaths


def train_dnn(training_set, settings):
    """
    """
    seed = np.random.seed(int(time.time()))
    pid = training_set[:, -1]
    features = training_set[:, :-1]
    print(np.shape(features))
    print(pid)

    neurons = settings.layers

    if settings.batchnorm:
        bnorm_layer = Normalization(axis=1, mean=0, variance=10)
        # layer.adapt(features)
        features = bnorm_layer(features)

    if not settings.dropout == 0:
        dr_layer = AlphaDropout(settings.dropout, seed=seed)
        features = dr_layer(features, training=True)

    optimizer = Adam(learning_rate=settings.learning_rate)
    inputlayer = Input(shape=(np.shape(features)[1],))
    hiddenlayer = Dense(neurons[0], activation='relu')(inputlayer)
    for i in neurons[1:]:
        hiddenlayer = Dense(i, activation='relu')(hiddenlayer)
    outputlayer = Dense(1, activation='sigmoid')(hiddenlayer)
    deepnn = Model(inputs=inputlayer, outputs=outputlayer)
    deepnn.compile(loss='binary_crossentropy', optimizer=optimizer)
    deepnn.summary()

    history = deepnn.fit(features, pid, validation_split=0.5,
                         epochs=settings.epochnum, verbose=settings.verbose,
                         batch_size=settings.batch_size)

    if settings.showhistory:
        plt.figure('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Binary CrossEntropy Loss')

        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.legend()
        plt.savefig(os.path.join('fig', "epochs.pdf"))
        plt.show()

    return deepnn


def eval_dnn(dnn, eval_set, plot_opt=[], flag_data=True):
    """
    """
    prediction_array = dnn.predict(
        eval_set) if flag_data else dnn.predict(eval_set[:, :-1])
    prediction_array = prediction_array.flatten()
    if plot_opt:
        plotname = plot_opt[0]
        plt.figure(plotname)
        plt.hist(prediction_array, bins=300, histtype='step',
                 color=plot_opt[1], label=plot_opt[2])
        plt.xlabel('y')
        plt.ylabel('Events per 1/300')  # MAKE IT BETTER
        plt.legend()
        plt.savefig('./fig/predict_'+plotname+'.pdf')
        plt.draw()

    return prediction_array


def dnn(txt_names, settings, txt_path='../data/txt', f_print=True):
    """
    """
    print(settings.layers)
    train_array_path, data_array_path = get_txtpaths(
        filenames=txt_names, rel_path=txt_path)

    training_set = np.loadtxt(train_array_path)
    pi_set = np.array([training_set[i, :] for i in range(
        np.shape(training_set)[0]) if training_set[i, -1] == 0])
    K_set = np.array([training_set[i, :] for i in range(
        np.shape(training_set)[0]) if training_set[i, -1] == 1])
    data_set = np.loadtxt(data_array_path)

    deepnn = train_dnn(training_set, settings)

    pi_eval = eval_dnn(deepnn, pi_set, flag_data=False,
                       plot_opt=['Templ_eval', 'red', 'Evaluated pions'])
    K_eval = eval_dnn(deepnn, K_set, flag_data=False,
                      plot_opt=['Templ_eval', 'blue', 'Evaluated kaons'])
    pred_array = eval_dnn(deepnn, data_set, flag_data=True,
                          plot_opt=['Dataeval', 'blue', 'Evaluated data'])

    if f_print:
        f_pred = np.sum(pred_array)
        print(f'The predicted K fraction is : {f_pred/len(pred_array)}')
        print('Max prediction :', np.max(pred_array))
        print('Min prediction :', np.min(pred_array))

    return pi_eval, K_eval, pred_array


if __name__ == '__main__':

    input_files = ['train_array_prova.txt', 'data_array_prova.txt']

    settings = dnn_settings()
    settings.layers = [60, 45, 30, 15]
    settings.batch_size = 128
    settings.epochnum = 200
    settings.verbose = 2
    settings.batchnorm = False
    # settings.dropout = 0.005
    settings.learning_rate = 5e-5

    pi_eval, K_eval, pred_array = dnn(input_files, settings)
    # plt.show()
