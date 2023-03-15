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
from utilities.dnn_settings import DnnSettings
from utilities.utils import default_rootpaths, default_txtpaths, default_vars, \
                            find_cut, roc, default_figpath
from utilities.exceptions import InvalidSourceError


def train_dnn(training_set, settings, savefig=True, figname='history',
              trained_filenames=('deepnn.json', 'deepnn.h5')):
    """
    Trains a Keras deep neural network.

    :param training_set: 2D numpy array with flag {0,1} as last column for training the DNN.
    :type training_set: 2D numpy.array[float]
    :param settings: DnnSettings instance with the settings for the generation of the DNN.
    :type settings: utilities.import_datasets.DnnSettings class instance
    :param savefig: If ``True``, saves the history plot of the training.
    :type savefig: bool
    :param figname: If savefig=``True``, saves the history figure with this name.
    :type figname: str
    :param trained_filenames: Two element tuple containing respectively the name of the .json of the file where to save the DNN and the .h5 where to store its weigths.
    :type trained_filenames: tuple[str]
    :return: Trained deep neural network.
    :rtype: keras.models.Model

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
        plt.savefig(default_figpath(figname))

    model_json = deepnn.to_json()
    with open(trained_filenames[0], "w") as json_file:
        json_file.write(model_json)
    deepnn.save_weights(trained_filenames[1])

    return deepnn


def eval_dnn(dnn, eval_set, flag_data=True,
             savefig=True, plot_opt=[], figname=""):
    """
    Evaluates a given keras deep neural network on a given dataset and
    optionally plots the results.

    :param dnn: Trained deep neural network to be used in the evaluation.
    :type dnn: keras.models.Model
    :param eval_set: 2D numpy array to evaluate.
    :type eval_set: 2D numpy.array[float]
    :param flag_data: If ``True`` assumes the eval_set array has no flag column at the end, otherwise it stripsa away the last column before evaluation.
    :type flag_data: bool
    :param savefig: If ``True``, saves the histogram of the evaluation results of the given set.
    :type savefig: bool
    :param plot_opt: Three-element list, containing the name of the plot, the color of the histogram and its label (in this order).
    :type plot_opt: list[str]
    :param figname: If savefig=``True``, saves the evaluation figure with this name.
    :type figname: str
    :return: Predictions of the events (rows) of the eval_set.
    :rtype: numpy.array[float]

    """
    prediction_array = dnn.predict(eval_set).flatten() \
        if flag_data else dnn.predict(eval_set[:, :-1]).flatten()

    if savefig and len(plot_opt) == 3:  # !!!! MAKE IT BETTER (E.G. KWARGS)
        nbins = 300
        plotname = plot_opt[0]
        plt.figure(plotname)
        plt.hist(prediction_array, bins=nbins, histtype='step',
                 color=plot_opt[1], label=plot_opt[2])
        plt.xlabel('y')
        plt.ylabel(f'Events per 1/{nbins}')
        plt.yscale('log')
        plt.legend()
        if figname == '':
            plt.savefig(default_figpath('predict_'+plotname))
        else:
            plt.savefig(figname)
        plt.draw()

    return prediction_array


def dnn(source=('root', default_rootpaths()), root_tree='tree;1',
        vars=default_vars(), n_mc=560000, n_data=50000, settings=DnnSettings(),
        load=False, trained_model=('deepnn.json', 'deepnn.h5'),
        savefigs=False, fignames=("", "", "", "")):
    """
    Trains or loads a deep neural network and evaluates it on the training sets used to train it, as well as a ulterior dataset.

    :param source: Two element tuple containing the options for how to build the datasets for the DNN and the relative paths. The first item can be either 'txt' or 'root'. In case it is built from txt, the second element of source must be a tuple containing two .txt paths, one relative to the training set and the other to the set to be evaluated. The .txt files must be in a format compatible with numpy's loadtxt() and savetxt() methods. In case it is built from root the second element of source must be a tuple containing three .root file paths, containing the "background" sample (flag=0), the "signal" one (flag=1) and the mixed one, in this order.
    :type source: tuple[{'root','txt'},tuple[str]]
    :param root_tree: In case of 'root' source, the name of the tree from which to load variables.
    :type root_tree: str
    :param vars: In case of 'root' source, tuple containing the names of the features to load for the DNN.
    :type vars: tuple[str]
    :param n_mc: In case of 'root' source, number of events to take from the root files for the training set.
    :type n_mc: int
    :param n_data: In case of 'root' source, number of events to take from the root file for the mixed set.
    :type n_data: int
    :param settings: DnnSettings instance with the settings for the generation of the DNN.
    :type settings: utilities.import_datasets.DnnSettings class instance
    :param load: If ``True``, instead of training a new DNN, it loads it from the files given in "trained_model".
    :type load: bool
    :param trained_filenames: Two element tuple containing respectively the name of the .json and .h5 files from which to load the DNN structure and weights, if load=``True``.
    :type trained_filenames: tuple[str]
    :param savefigs: If ``True``, saves the training history (if the DNN was not loaded) and the histograms of the evaluation results on the training and mixed datasets.
    :type savefigs: bool
    :param fignames: Four element tuple containing the figure names (in case savefigs=``True``) for: DNN training history figure, evaluated training species figures, evaluated mixed dataset figure.
    :type fignames: tupel[str]
    :return: Tuple of numpy arrays, containing the evaluated background set, the evaluated signal set and the evaluated data set (in this order).
    :rtype: tuple[numpy.array[float]]

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
            training_set, settings, savefig=savefigs, figname=fignames[0],
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
                       figname=fignames[1])
    k_eval = eval_dnn(deepnn, k_set, flag_data=False, savefig=savefigs,
                      plot_opt=['Templ_eval', 'blue', 'Evaluated kaons'],
                      figname=fignames[2])
    pred_array = eval_dnn(deepnn, data_set, flag_data=True, savefig=savefigs,
                          plot_opt=['Data_eval', 'blue', 'Evaluated data'],
                          figname=fignames[3])

    return pi_eval, k_eval, pred_array


if __name__ == '__main__':

    settings = DnnSettings(layers=(75, 60, 45, 30, 20),
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
