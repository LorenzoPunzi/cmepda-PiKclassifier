"""
Module containing the functions to train a DNN with multiple variables
(features) in numpy arrays, to perform an evaluation on a given dataset and to
apply them the algorithm that estimates the fraction of Kaons present in a
mixed dataset
"""
import traceback
import time
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, AlphaDropout
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from utilities.import_datasets import array_generator
from utilities.dnn_settings import DnnSettings
from utilities.utils import default_rootpaths, default_txtpaths, default_vars,\
                            default_figpath, find_cut, stat_error, syst_error
from utilities.exceptions import InvalidSourceError, IncorrectEfficiencyError,\
                                 IncorrectIterableError

warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'


def train_dnn(training_set, settings, savefig=True, figname='',
              trained_filenames=('deepnn.json', 'deepnn.h5')):
    """
    Trains a Keras deep neural network.

    :param training_set: 2D numpy array with flag={0,1} as last column for training the DNN.
    :type training_set: 2D numpy.array[float]
    :param settings: DnnSettings instance with the settings for the generation of the DNN.
    :type settings: utilities.import_datasets.DnnSettings class instance
    :param savefig: If ``True``, saves the history plot of the training.
    :type savefig: bool
    :param figname: If ``savefig==True``, saves the loss history figure with this name.
    :type figname: str
    :param trained_filenames: Two element tuple containing respectively the name of the .json of the file where to save the DNN and the .h5 where to store its weigths.
    :type trained_filenames: tuple[str]
    :return: Trained deep neural network.
    :rtype: keras.models.Model

    """
    seed = np.random.seed(int(time.time()))
    pid = training_set[:, -1]
    features = training_set[:, :-1]

    

    neurons = settings.layers

    # Sets the dropout probability to pass in a keras AlphaDropout layer
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

    if savefig is True:
        plt.figure('Losses')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Binary CrossEntropy Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.legend()
        if figname == '':
            plt.savefig(default_figpath('History'))
        else:
            plt.savefig(figname)

    # Model and weights are saved in the apposite files
    model_json = deepnn.to_json()
    with open(trained_filenames[0], "w") as json_file:
        json_file.write(model_json)
    deepnn.save_weights(trained_filenames[1])

    return deepnn


def eval_dnn(dnn, eval_set, flag_data=True,
             savefig=True, plot_opt=[], figname=""):
    """
    Evaluates a given keras deep neural network on a given dataset.

    :param dnn: Trained deep neural network to be used in the evaluation.
    :type dnn: keras.models.Model
    :param eval_set: 2D numpy array to evaluate.
    :type eval_set: 2D numpy.array[float]
    :param flag_data: If ``True`` assumes the eval_set array has no flag column at the end, otherwise it strips away the last column before evaluation.
    :type flag_data: bool
    :param savefig: If ``True`` saves the histogram of the evaluation results of the given set.
    :type savefig: bool
    :param plot_opt: Three-element tuple or list, containing the name of the plot, the color of the histogram and its label (in this order).
    :type plot_opt: tuple[str] or list[str]
    :param figname: If savefig=``True``, saves the evaluation figure with this name.
    :type figname: str
    :return: Predictions of the of the eval_set's events.
    :rtype: numpy.array[float]

    """
    prediction_array = dnn.predict(eval_set).flatten() \
        if flag_data else dnn.predict(eval_set[:, :-1]).flatten()

    
    try:
        if (type(plot_opt) is not list and type(plot_opt) is not tuple):
            raise IncorrectIterableError(plot_opt, 3, 'plot_opt')
        elif len(plot_opt) < 3:
            raise IncorrectIterableError(plot_opt, 3, 'plot_opt')
    except IncorrectIterableError:
        print(traceback.format_exc())
        sys.exit()
    if len(plot_opt) >= 4:
        msg = '***WARNING*** \nPlot options given are more than three. Using\
only the first three...\n*************\n'
        warnings.warn(msg, stacklevel=2)
        plot_opt = plot_opt[:3]

    if savefig is True:
        nbins = 300
        plotname = plot_opt[0]
        plt.figure(plotname)
        plt.hist(prediction_array, bins=nbins, histtype='step',
                 color=plot_opt[1], label=plot_opt[2])
        plt.title(plotname)
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
        vars=default_vars(), n_mc=560000, n_data=50000, test_split=0.2,
        settings=DnnSettings(), efficiency=0.9, error_optimization=True,
        load=False, trained_filenames=('deepnn.json', 'deepnn.h5'),
        savefigs=False, figpath='', fignames=("", "", "", "")):
    """
    Trains or loads a deep neural network and uses it to estimate the fraction
    \'f\' of Kaons in the mixed dataset. To do that, an evaluation of the dnn
    on the "testing dataset" with a fixed value of efficiency is required

    :param source: Two element tuple containing the options for how to build the datasets for the DNN and the relative paths. The first item can be either 'txt' or 'root'. In case it is built from txt, the second element of source must be a tuple containing two .txt paths, one relative to the training set and the other to the data set to be evaluated. The .txt files must be in a format compatible with numpy's loadtxt() and savetxt() methods. In case it is built from root the second element of source must be a tuple containing three .root file paths, for the "background" set (flag=0), the "signal" set (flag=1) and the data one, in this order.
    :type source: tuple[{'root','txt'}, tuple[str]]
    :param root_tree: In case of 'root' source, the name of the tree from which to load variables.
    :type root_tree: str
    :param vars: In case of 'root' source, tuple containing the names of the features to load for the DNN.
    :type vars: tuple[str]
    :param n_mc: In case of 'root' source, number of events to take from the root files for the training set.
    :type n_mc: int
    :param n_data: In case of 'root' source, number of events to take from the root file for the mixed set.
    :type n_data: int
    :param test_split: Fraction of the training array that is used for the testing
    :type test_split: float
    :param settings: DnnSettings instance with the settings for the generation of the DNN.
    :type settings: utilities.import_datasets.DnnSettings class instance
    :param load: If ``True``, instead of training a new DNN, it loads it from the files given in "trained_model".
    :type load: bool
    :param trained_filenames: Two element tuple containing respectively the name of the .json and .h5 files from which to load the DNN structure and weights, if ``load==True``.
    :type trained_filenames: tuple[str]
    :param efficiency: Sensitivity required from the test
    :type efficiency: float
    :param error_optimization: Performs error optimization instead of using a fixed efficiency value
    :type error_optimization: bool
    :param savefigs: If ``True``, saves the training history (if the DNN was not loaded) and the histograms of the evaluation results on the training and mixed datasets.
    :type savefigs: bool
    :param figpath: Path where to save the figures (in case ``savefigs==True``)
    :type figpath: str
    :param fignames: Four element tuple containing the figure names (in case ``savefigs==True``) for: DNN training history figure, evaluated training species figures, evaluated mixed dataset figure.
    :type fignames: tuple[str]
    :return: Estimated signal fraction (with uncertainties), parameters of the test algorithm and arrays containing the DNN evaluation of the testing array (divided for the two species)
    :rtype: tuple[float], tuple[float], tuple[numpy.array[float]]

    """

    try:
        if (type(source) is not list and type(source) is not tuple):
            raise IncorrectIterableError(source, 2, 'source')
        elif len(source) < 2:
            raise IncorrectIterableError(source, 2, 'source')
    except IncorrectIterableError:
        print(traceback.format_exc())
        sys.exit()
    if len(source) >= 3:
        msg = f'***WARNING*** \nInput source given is longer than two.\
Using only the first two...\n*************\n'
        warnings.warn(msg, stacklevel=2)
        source = source[:2]

    
    try:
        if source[0] == 'txt':
            try:
                if (type(source[1]) is not list and type(source[1]) is not tuple):
                    raise IncorrectIterableError(source[1], 2, 'paths')
                elif len(source[1]) < 2:
                    raise IncorrectIterableError(source[1], 2, 'paths')
            except IncorrectIterableError:
                print(traceback.format_exc())
                sys.exit()
            if len(source[1]) >= 3:
                msg = f'***WARNING*** \nInput source paths given are more than two.\
Using only the first two...\n*************\n'
                warnings.warn(msg, stacklevel=2)
                source[1] = source[1][:3]
            mc_array_path, data_array_path = source[1] if source[1] \
                else default_txtpaths()
            mc_set, data_set = np.loadtxt(mc_array_path), \
                np.loadtxt(data_array_path)
        elif source[0] == 'root':
            try:
                if (type(source[1]) is not list and type(source[1]) is not tuple):
                    raise IncorrectIterableError(source[1], 3, 'paths')
                elif len(source[1]) < 3:
                    raise IncorrectIterableError(source[1], 3, 'paths')
            except IncorrectIterableError:
                print(traceback.format_exc())
                sys.exit()
            if len(source[1]) >= 4:
                msg = f'***WARNING*** \nInput source paths given are more than three.\
Using only the first three...\n*************\n'
                warnings.warn(msg, stacklevel=2)
                source[1] = source[1][:3]
            mc_set, data_set = array_generator(rootpaths=source[1],
                                               tree=root_tree, vars=vars,
                                               n_mc=n_mc, n_data=n_data)
        else:
            raise InvalidSourceError(source[0])
    except InvalidSourceError:
        print(traceback.format_exc())
        sys.exit()

    try:
        if type(efficiency) is not float:
            raise IncorrectEfficiencyError(efficiency)
        elif efficiency<=0 or efficiency>=1:
            raise IncorrectEfficiencyError(efficiency)
    except IncorrectEfficiencyError:
        print(traceback.format_exc())
        sys.exit()

    num_mc = len(mc_set[:, 0])

    # The original "training" set is split in two, one part for training and
    # validation, the other for testing. The fraction of the testing array
    # w.r.t. the initial one is equal the "test_split" value
    training_set = mc_set[:int((1-test_split)*num_mc), :]
    test_set = mc_set[int((1-test_split)*num_mc):-1, :]

    module_path = os.path.dirname(__file__)

    # Training of the neural network
    if load is not True:
        deepnn = train_dnn(
            training_set, settings, savefig=savefigs, figname=f'{figpath}/{fignames[0]}',
            trained_filenames=trained_filenames)
        np.savetxt(f'{module_path}/testing_array.txt', test_set)
    else:
        test_set = np.loadtxt(f'{module_path}/testing_array.txt')
        json_path = trained_filenames[0]
        weights_path = trained_filenames[1]
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        deepnn = model_from_json(loaded_model_json)
        deepnn.load_weights(weights_path)
        deepnn.summary()

    pi_test = np.array([test_set[i, :] for i in range(
        np.shape(test_set)[0]) if test_set[i, -1] == 0])
    k_test = np.array([test_set[i, :] for i in range(
        np.shape(test_set)[0]) if test_set[i, -1] == 1])

    # Evaluation of the dnn on the testing dataset and on data
    pi_eval = eval_dnn(deepnn, pi_test, flag_data=False, savefig=savefigs,
                       plot_opt=['Templ_eval', 'red', 'Evaluated pions'],
                       figname=f'{figpath}/{fignames[1]}')
    k_eval = eval_dnn(deepnn, k_test, flag_data=False, savefig=savefigs,
                      plot_opt=['Templ_eval', 'blue', 'Evaluated kaons'],
                      figname=f'{figpath}/{fignames[2]}')

    data_eval = eval_dnn(deepnn, data_set, flag_data=True, savefig=savefigs,
                         plot_opt=['Data_eval', 'blue', 'Evaluated data'],
                         figname=f'{figpath}/{fignames[3]}')

    used_eff = 0
    df_opt = -99999

    if error_optimization is True:  # Enables FOM maximization
        efficiencies = np.linspace(0.25, 0.999, 200)
        for tmp_eff in efficiencies:
            tmp_cut, tmp_misid = find_cut(pi_eval, k_eval, tmp_eff)
            tmp_frac = ((data_eval > tmp_cut).sum()/data_eval.size
                        - tmp_misid)/(tmp_eff-tmp_misid)
            tmp_dfopt = -np.sqrt(stat_error(tmp_frac, data_eval.size, tmp_eff, tmp_misid)
                                 ** 2+syst_error(tmp_frac, (pi_eval.size, k_eval.size), tmp_eff, tmp_misid)**2)
            if tmp_dfopt >= df_opt:
                df_opt = tmp_dfopt
                used_eff = tmp_eff
    else:
        used_eff = efficiency  # Value of efficiency effectively used for the analysis

    cut, misid = find_cut(pi_eval, k_eval, used_eff)

    if savefigs:
        plt.figure('Templ_eval')
        plt.axvline(x=cut, color='green', label=f'y cut for eff={used_eff}')
        plt.legend()
        plt.savefig(f'{figpath}/eval_Templates.png')

    fraction = ((data_eval > cut).sum()/data_eval.size-misid)/(used_eff-misid)

    df_stat = stat_error(fraction, data_eval.size, used_eff, misid)

    df_syst = syst_error(
        fraction, (pi_eval.size, k_eval.size), used_eff, misid)

    print(f'y cut is {cut} for {used_eff} efficiency\n')
    print(
        f'Misid is {misid} +- {np.sqrt(misid*(1-misid)/pi_eval.size)} for {used_eff} efficiency\n')
    print(
        f'The estimated fraction of K events is {fraction} +- {df_stat} (stat) +- {df_syst} (syst)\n')

    fr = (fraction, df_stat, df_syst)

    algorithm_parameters = (used_eff, misid, cut)
    eval_test = (pi_eval, k_eval)

    return fr, algorithm_parameters, eval_test


if __name__ == '__main__':
    print('Running this module as main module is not supported. Feel free to \
add a custom main or run the package as a whole (see README.md)')
