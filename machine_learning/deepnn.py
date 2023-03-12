"""
Trains a DNN with a numpy array with variable data columns to
distinguish between pions and Kaons given multiple variables (features)
on which to train simultaneously
"""

import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Normalization, AlphaDropout
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from utilities.import_datasets import array_generator
from utilities.dnn_settings import dnn_settings
from utilities.utils import default_rootpaths, default_txtpaths, default_vars, \
                            find_cut, roc, default_figpath
from utilities.exceptions import InvalidSourceError


def train_dnn(training_set, settings, savefig=True, figpath="",
              trained_filenames=('deepnn.json', 'deepnn.h5')):
    """
    """
    seed = np.random.seed(int(time.time()))
    pid = training_set[:, -1]
    features = training_set[:, :-1]
    # print(np.shape(features))
    # print(pid)

    neurons = settings.layers

    if settings.dropout != 0:
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

    history = deepnn.fit(features, pid, validation_split=settings.val_fraction,
                         epochs=settings.epochnum, verbose=settings.verbose,
                         batch_size=settings.batch_size)

    if savefig:
        plt.figure('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Binary CrossEntropy Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.legend()
        if figpath == "":
            plt.savefig(default_figpath('history'))
        else:
            plt.savefig(figpath)

    model_json = deepnn.to_json()
    with open(trained_filenames[0], "w") as json_file:
        json_file.write(model_json)
    deepnn.save_weights(trained_filenames[1])

    return deepnn


def eval_dnn(dnn, eval_set, plot_opt=[], flag_data=True, savefig=True, figpath=""):
    """
    """
    prediction_array = dnn.predict(eval_set).flatten() \
        if flag_data else dnn.predict(eval_set[:, :-1]).flatten()

    if savefig and len(plot_opt) == 3:  # !!!! MAKE IT BETTER
        nbins = 300
        plotname = plot_opt[0]
        plt.figure(plotname)
        plt.hist(prediction_array, bins=nbins, histtype='step',
                 color=plot_opt[1], label=plot_opt[2])
        plt.xlabel('y')
        plt.ylabel(f'Events per 1/{nbins}')  # MAKE IT BETTER
        plt.yscale('log')
        plt.legend()
        if figpath == "":
            plt.savefig(default_figpath('predict_'+plotname))
        else:
            plt.savefig(figpath)
        plt.draw()

    return prediction_array


def dnn(source=('root', default_rootpaths()), root_tree='t_M0pipi;1',
        vars=default_vars(), n_mc=560000, n_data=50000, settings=dnn_settings(),
        load=False, trained_model=('deepnn.json', 'deepnn.h5'), stat_split=0,
        savefigs=False, figpaths=("", "", "", "")):
    """
    """
    try:
        if source[0] == 'txt':
            mc_array_path, data_array_path = source[1] if source[1] \
                else default_txtpaths()
            training_set = np.loadtxt(mc_array_path)
            data_set = np.loadtxt(data_array_path)
        elif source[0] == 'root':
            training_set, data_set = array_generator(rootpaths=source[1],
                                                     tree=root_tree, vars=vars,
                                                     n_mc=n_mc, n_data=n_data)
        else:
            raise InvalidSourceError(source[0])
    except InvalidSourceError as err:
        print(err)
        sys.exit()

    pi_set = np.array([training_set[i, :] for i in range(
        np.shape(training_set)[0]) if training_set[i, -1] == 0])
    k_set = np.array([training_set[i, :] for i in range(
        np.shape(training_set)[0]) if training_set[i, -1] == 1])

    if load is not True:
        deepnn = train_dnn(
            training_set, settings, savefig=savefigs, figpath=figpaths[0],
            trained_filenames=trained_model)
    else:
        json_path = trained_model[0]
        weights_path = trained_model[1]
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        deepnn = model_from_json(loaded_model_json)
        deepnn.load_weights(weights_path)
        deepnn.summary()

    pi_eval = eval_dnn(deepnn, pi_set, flag_data=False, savefig=savefigs,
                       plot_opt=['Templ_eval', 'red', 'Evaluated pions'],
                       figpath=figpaths[1])
    k_eval = eval_dnn(deepnn, k_set, flag_data=False, savefig=savefigs,
                      plot_opt=['Templ_eval', 'blue', 'Evaluated kaons'],
                      figpath=figpaths[2])
    pred_array = eval_dnn(deepnn, data_set, flag_data=True, savefig=savefigs,
                          plot_opt=['Data_eval', 'blue', 'Evaluated data'],
                          figpath=figpaths[3])

    return pi_eval, k_eval, pred_array


if __name__ == '__main__':

    settings = dnn_settings(layers=(75, 60, 45, 30, 20),
                            batch_size=128,
                            epochnum=10,
                            learning_rate=5e-5)

    pi_eval, k_eval, data_eval = dnn(settings=settings, load_trained=True)
    efficiency = 0.95

    y_cut, misid = find_cut(pi_eval, k_eval, efficiency)
    plt.axvline(x=y_cut, color='green', label='y cut for '
                + str(efficiency)+' efficiency')
    plt.legend()
    plt.savefig(default_figpath('ycut'))

    rocdnnx, rocdnny, aucdnn = roc(pi_eval, k_eval, eff=efficiency,
                                   inverse_mode=False, makefig=True,
                                   name="dnn_roc")

    '''
    rocvarcutx, rocvarcuty, aucvarcut = var_cut(drawfig=False, draw_roc=False)
    _, dtcy, dtcx = dt_classifier(print_tree='')

    print(dtcx, dtcy)

    plot_rocs(rocx_array=[rocdnnx, rocvarcutx],
              rocy_array=[rocdnny, rocvarcuty], auc_array=[aucdnn, aucvarcut],
              inverse_mode_array=(False, True),
              roc_labels=('deep nn', 'cut on M0_Mpipi'),
              roc_linestyles=('-', '-'), roc_colors=('red', 'green'),
              x_pnts=(dtcx,), y_pnts=(dtcy,),
              point_labels=('decision tree classifier',))
    '''

    print(f'y cut is {y_cut} , misid is {misid}')
    f = ((data_eval > y_cut).sum()/data_eval.size-misid)/(efficiency-misid)
    print(f'The estimated fraction of K events is {f}')

    plt.show()
