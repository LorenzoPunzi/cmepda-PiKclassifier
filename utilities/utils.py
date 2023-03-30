"""
Module containing general-use functions
"""
import traceback
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from utilities.exceptions import IncorrectEfficiencyError, \
                                 IncorrectIterableError, IncoherentRocPlotError

warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'


def default_rootpaths():
    """
    Returns the default root file paths of the package, where background is
    pions and signal is kaons.

    :return: Three element tuple containing the paths of the pion MC, the kaon MC and the mixed data root files, respectively.
    :rtype: tuple[str]

    """
    current_path = os.path.dirname(__file__)
    rel_path = '../data/root_files'
    rootnames = ['B0PiPi_MC.root', 'B0sKK_MC.root', 'Bhh_data.root']
    rootpaths = tuple([os.path.join(current_path, rel_path, file)
                       for file in rootnames])
    return rootpaths


def default_txtpaths():
    """
    Returns the .txt file paths containing the training MC array and the data
    array, to be used in DNN or DTC analyses.

    :return: Tuple containing the paths of the MC training array (50/50 signal/background for unbiased training) and the path of the data array, respectively.
    :rtype: tuple[str]

    """
    current_path = os.path.dirname(__file__)
    rel_path = '../data/txt'
    txtnames = ['train_array.txt', 'data_array.txt']
    txtpaths = tuple([os.path.join(current_path, rel_path, file)
                      for file in txtnames])
    return txtpaths


def default_vars():
    """
    Returns default variables used by the package in the pi-K analysis.

    :return: 13 element tuple containing the names of the default variables to use.
    :rtype: tuple[str]

    """
    return ('M0_Mpipi', 'M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_p', 'M0_pt',
            'M0_eta', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2',
            'h2_thetaC0', 'h2_thetaC1', 'h2_thetaC2')


def default_figpath(figname, dir='fig', figtype='pdf'):
    """
    Returns the path to the figure folder with respect to the cwd in which to
    then save figures.

    :param figname: Name with which to save the figure.
    :type figname: str
    :param dir: Directory relative to cwd where to save the figure.
    :type dir: str
    :param figtype: Type of figure save file.
    :type figtype: str
    :return: Path where to save the figure.
    :rtype: str

    """
    wd_path = os.getcwd()
    figpath = os.path.join(wd_path, dir, figname+'.'+figtype)
    return figpath


def default_resultsdir(dir='outputs-PiKclassifier'):
    """
    Returns the path where to store the outputs of the package.

    :param dir: Directory where to save the outputs.
    :type dir: str
    :return: The figure path.
    :rtype: str

    """
    if os.path.exists(os.getcwd()+'/'+dir):
        pass
    else:
        os.mkdir(dir)
    figpath = os.path.join(os.getcwd(), dir)
    return figpath


def find_cut(pi_array, k_array, efficiency,
             specificity_mode=False, inverse_mode=False):
    """
    Finds where to cut a certain variable to obtain a certain
    sensitivity/specificity in a hypothesis test between two given species' arrays.

    :param pi_array: Array containing the background species.
    :type pi_array: numpy.array[float]
    :param k_array: Array containing the signal species.
    :type k_array: numpy.array[float]
    :param efficiency: Sensitivity required from the test (specificity if ``specificity_mode = True``).
    :type efficiency: float
    :param specificity_mode: If set to ``True`` the efficiency given is taken to be the intended specificity.
    :type specificity_mode: bool
    :param inverse_mode: Set to ``True`` if the signal events tend to have lower values.
    :type inverse_mode: bool
    :return: Two element tuple containing cut value and misidentification probability for the negative species (or sensitivity if ``specificity_mode = True``)
    :rtype: tuple[double]
"""
    

    if inverse_mode:
        efficiency = 1 - efficiency
        cut = - np.sort(-k_array)[int(efficiency*(len(k_array)-1))] \
            if not specificity_mode else np.sort(pi_array)[int(efficiency*(len(k_array)-1))]
        misid = (pi_array < cut).sum()/pi_array.size \
            if not specificity_mode else (k_array < cut).sum()/k_array.size
    else:
        cut = - np.sort(-k_array)[int(efficiency*(len(k_array)-1))] \
            if not specificity_mode else np.sort(pi_array)[int(efficiency*(len(k_array)-1))]
        misid = (pi_array > cut).sum()/pi_array.size \
            if not specificity_mode else (k_array > cut).sum()/k_array.size

    return cut, misid


def plot_rocs(rocx_arrays, rocy_arrays, roc_labels, roc_linestyles, roc_colors,
              x_pnts=(), y_pnts=(), point_labels=(''), eff=0,
              figtitle='ROC', figname=''):
    """
    Draws superimposed roc curves and/or points

    :param rocx_arrays: List or tuple of numpy arrays, each containing the respective x points of different roc curves to be plotted.
    :type rocx_arrays: list[numpy.array[float]] or tuple[numpy.array[float]]
    :param rocy_arrays: List or tuple of numpy arrays, each containing the respective y points of different roc curves to be plotted.
    :type rocy_arrays: list[numpy.array[float]] or tuple[numpy.array[float]]
    :param roc_labels: Names of the respective species whose roc coordinates were given.
    :type roc_labels: list[str] or tuple[str]
    :param roc_linestyles: Linestyles of the respective species whose roc coordinates were given.
    :type roc_linestyles: list[str] or tuple[str]
    :param roc_colors: Colors of the respective species whose roc coordinates were given.
    :type roc_colors: list[str] or tuple[str]
    :param x_pnts: List or tuple of the respective x coordinates of points to be plotted.
    :type x_pnts: list[double] or tuple[double]
    :param y_pnts: List or tuple of the respective y coordinates of points to be plotted.
    :type y_pnts: list[double] or tuple[double]
    :param point_labels: List or tuple of names of the respective species whose point coordinates were given.
    :type point_labels: list[str] or tuple[str]
    :param eff: If different than 0., draws a green dashed line at y = eff on the plot.
    :type eff: double
    :param figtitle: Title to be given to the figure.
    :type figtitle: str
    :param figname: If different than '', saves the figure as a pdf with name figname.
    :type figname: str

    """
    plt.figure(figtitle)
    plt.title(figtitle)
    plt.xlabel('False Positive Probability')
    plt.xlim(0, 1)
    plt.ylabel('True Positive Probability')
    plt.ylim(0, 1)

    try:
        if (type(rocx_arrays) == list or type(rocx_arrays) == tuple) is not True:
            raise IncorrectIterableError(rocx_arrays, 3, 'rocx_arrays')
    except IncorrectIterableError:
        print(traceback.format_exc())
        sys.exit()
    try:
        if (type(rocy_arrays) == list or type(rocy_arrays) == tuple) is not True:
            raise IncorrectIterableError(rocy_arrays, 3, 'rocy_arrays')
    except IncorrectIterableError:
        print(traceback.format_exc())
        sys.exit()
    try:
        if (type(roc_labels) == list or type(roc_labels) == tuple) is not True:
            raise IncorrectIterableError(roc_labels, 3, 'roc_labels')
    except IncorrectIterableError:
        print(traceback.format_exc())
        sys.exit()
    try:
        if (type(roc_linestyles) == list or type(roc_linestyles) == tuple) is not True:
            raise IncorrectIterableError(roc_linestyles, 3, 'roc_linestyles')
    except IncorrectIterableError:
        print(traceback.format_exc())
        sys.exit()
    try:
        if (type(roc_colors) == list or type(roc_colors) == tuple) is not True:
            raise IncorrectIterableError(roc_colors, 3, 'roc_colors')
    except IncorrectIterableError:
        print(traceback.format_exc())
        sys.exit()

    # Check if all the lists/tuples have same lengths
    try:
        if len(set([len(i) for i in [rocx_arrays, rocy_arrays, roc_labels, roc_linestyles, roc_colors]])) != 1:
            raise IncoherentRocPlotError
    except IncoherentRocPlotError:
        print(traceback.format_exc())
        sys.exit()

    for idx in range(len(rocx_arrays)):
        plt.plot(rocx_arrays[idx], rocy_arrays[idx], label=roc_labels[idx],
                 color=roc_colors[idx], linestyle=roc_linestyles[idx])
    for idx in range(len(x_pnts)):
        plt.plot((x_pnts[idx]), (y_pnts[idx]),
                 label=point_labels[idx], marker='o')

    if eff != 0:
        plt.axhline(y=eff, color='green', linestyle='--',
                    label='Efficiency chosen at ' + str(eff))
    plt.axline((0, 0), (1, 1), linestyle='--', label='AUC = 0.5')
    plt.legend()
    plt.draw()
    if figname == '':
        plt.savefig(default_figpath(figtitle))
    else:
        plt.savefig(figname)


def roc(pi_array, k_array, inverse_mode=False, makefig=False, eff=0, name="ROC"):
    """
    Returns the roc curve's x and y coordinates given two arrays of values for
    two different species. optionally draws the roc curve using plot_rocs().

    :param pi_array: Array containing the "negative" species.
    :type pi_array: numpy.array[float]
    :param k_array: Array containing the "positive" species.
    :type k_array: numpy.array[float]
    :param inverse_mode: To activate if the "positive" events tend to have lower values
    :type inverse_mode: bool
    :param makefig: If set to True draws the roc curve
    :type makefig: bool
    :param eff: If different than 0. and makefig = True , draws a green dashed line at y = eff on the plot.
    :type eff: double
    :param name: If makefig = True , name of the saved figure.
    :type name: str
    :return: Three element tuple containing: numpy array of floats of x coordinates of the roc curve, numpy array of floats of y coordinates of the roc curve, AUC of the ROC curve.
    :rtype: tuple[numpy.array[float], numpy.array[float], float]

    """
    true_array = np.concatenate(
        (np.zeros(pi_array.size), np.ones(k_array.size)))
    y_array = np.concatenate((pi_array, k_array))
    rocx, rocy, _ = metrics.roc_curve(true_array, y_array)
    auc = metrics.roc_auc_score(true_array, y_array)

    # need to invert the roc to make sense when in inverse mode
    if inverse_mode:
        rocx, rocy = np.ones(rocx.size)-rocx, np.ones(rocy.size)-rocy
        auc = 1 - auc

    if makefig:
        plot_rocs((rocx,), (rocy,), ("ROC",), ("-",), ("blue",), eff=eff,
                  figname=name)

    return rocx, rocy, auc


def stat_error(fraction, data_size, eff, misid):
    """
    Evaluates the statistical error on fraction estimate due to the finite
    sample of the data set, using the variance of sum of two binomials (of
    signal and background events respectively).

    :param fraction: Estimated fraction by the algorithm.
    :type fraction: float
    :param data_size: Size of the data set.
    :type template_sizes: int
    :param eff: Estimated efficiency of the algorithm.
    :type eff: float
    :param misid: Estimated misidentification probability (false positive) of the algorithm.
    :type misid: float
    :return: The statistical error associated to the fraction.
    :rtype: float

    """

    d_Nk = data_size*fraction*eff*(1-eff)
    d_Npi = data_size*(1-fraction)*misid*(1-misid)

    d_frac = np.sqrt(d_Nk+d_Npi)/(data_size*(eff-misid))

    return d_frac


def syst_error(fraction, template_sizes, eff, misid):
    """
    Evaluates the systematic error on fraction estimate due to the finite
    sample used to evaluate the "efficiency" and "misid" parameters.

    :param fraction: Estimated fraction by the algorithm.
    :type fraction: float
    :param template_sizes: Two element list or tuple of sizes of the evaluation arrays (background and signal dataset, in this order).
    :type template_sizes: list[int] tuple[int]
    :param eff: Estimated efficiency of the algorithm.
    :type eff: float
    :param misid: Estimated misidentification probability (false positive) of the algorithm
    :type misid: float
    :return: The systematic error associated to the fraction.
    :rtype: float

    """
    d_eff = np.sqrt(eff*(1-eff)/template_sizes[1])
    d_misid = np.sqrt(misid*(1-misid)/template_sizes[0])

    d_frac = np.sqrt((d_misid*(1-fraction))**2
                     + (d_eff*fraction)**2)/(eff-misid)

    return d_frac


if __name__ == '__main__':
    print('Running this module as main module is not supported. Feel free to \
add a custom main or run the package as a whole (see README.md)')
