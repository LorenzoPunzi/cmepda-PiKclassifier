import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics


def default_rootpaths():
    """
    Returns the default root file paths of the package.

    :return: Path tuple, containing the pion MC, the kaon Mc and the mixed data root files respectively.
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
    Returns the txt file paths of the package.

    :return: Path tuple, containing the pion MC, the kaon Mc and the mixed data txt files respectively.
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
    Returns default variables of the package.

    :return: Variable tuple.
    :rtype: tuple[str]

    """
    return ('M0_Mpipi', 'M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_p', 'M0_pt',
            'M0_eta', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2',
            'h2_thetaC0', 'h2_thetaC1', 'h2_thetaC2')


def default_figpath(figname, dir='fig', figtype='pdf'):
    """
    Returns the path to the figure folder with respect to the cwd in which to then save figures.

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
    Finds where to cut a certain varibale to obtain a certain sensitivity in a hypothesis test between two given species' arrays

    :param pi_array: Array containing the "negative" species.
    :type pi_array: numpy.array[float]
    :param k_array: Array containing the "positive" species.
    :type k_array: numpy.array[float]
    :param efficiency: Sensitivity required from the test (specificity in specificity mode)
    :type efficiency: float
    :param specificity_mode: To activate if the efficiency given is the specificity
    :type specificity_mode: bool
    :param inverse_mode: To activate if the "positive" events tend to have lower values
    :type inverse_mode: bool
    :return: Two element tuple containing cut value and misidentification probability for the negative species (or sensitivity in specificity mode)
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


def plot_rocs(rocx_array, rocy_array, roc_labels, roc_linestyles, roc_colors,
              x_pnts=(), y_pnts=(), point_labels=(''), eff=0,
              figtitle='ROC', figname=''):
    """
    Draws superimposed roc curves and/or points

    :param rocx_array: Array of arrays, each containing the respective x points of different roc curves to be plotted.
    :type rocx_array: numpy.array[numpy.array[float]]
    :param rocy_array: Array of arrays, each containing the respective y points of different roc curves to be plotted.
    :type rocy_array: numpy.array[numpy.array[float]]
    :param roc_labels: Names of the respective species whose roc coordinates were given.
    :type roc_labels: list[str] or tuple[str]
    :param roc_linestyles: Linestyles of the respective species whose roc coordinates were given.
    :type roc_linestyles: list[str] or tuple[str]
    :param roc_colors: Colors of the respective species whose roc coordinates were given.
    :type roc_colors: list[str] or tuple[str]
    :param x_pnts: Tuple containing the respective x coordinates of points to be plotted.
    :type x_pnts: list[double] or tuple[double]
    :param y_pnts: Tuple containing the respective y coordinates of points to be plotted.
    :type y_pnts: list[double] or tuple[double]
    :param point_labels: Names of the respective species whose point coordinates were given.
    :type point_labels: list[str] or tuple[str]
    :param eff: If different than 0., draws a green dashed line at y = eff on the plot.
    :type eff: double
    :param figtitle: Title to be given to the figure.
    :type figtitle: str
    :param figname: If different than '', saves the figure as a pdf with name figname.
    :type figname: str

    """
    plt.figure(figtitle)
    plt.xlabel('False Positive Probability')
    plt.xlim(0, 1)
    plt.ylabel('True Positive Probability')
    plt.ylim(0, 1)
    for idx in range(len(rocx_array)):
        '''
        flag_inverse = inverse_mode_array[idx]
        if flag_inverse is True:
            rocx_array[idx] = np.ones(rocx_array[idx].size) - rocx_array[idx]
            rocy_array[idx] = np.ones(rocy_array[idx].size) - rocy_array[idx]
            auc_array[idx] = 1 - auc_array[idx]
        '''
        plt.plot(rocx_array[idx], rocy_array[idx], label=roc_labels[idx],
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
    Returns the roc curve's x and y coordinates given two arrays of values for two different species. optionally draws the roc curve using plot_rocs().

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

    if inverse_mode:
        rocx, rocy = np.ones(rocx.size)-rocx, np.ones(rocy.size)-rocy
        auc = 1 - auc

    # print(f'AUC of the ROC is {auc}')

    if makefig:
        plot_rocs((rocx,), (rocy,), ("ROC",), ("-",), ("blue",), eff=eff,
                  figname=name)

    return rocx, rocy, auc


def syst_error(fraction, template_sizes, eff, misid):
    """
    Evaluates the systematic error on fraction estimate due to the finite sample
    used to evaluate the "efficiency" and "misid" parameters.

    :param fraction: Estimated fraction
    :type fraction: float
    :param template_sizes: Sizes of the evaluation arrays (pi and k dataset, in this order)
    :type template_sizes: tuple[int]
    :param eff: Efficiency of the algorithm
    :type eff: float
    :param misid: Misidentification probability (false positive) of the algorithm
    :type misid: float
    :return: The systematic error associated with the fraction
    :rtype: float

    """
    d_eff = np.sqrt(eff*(1-eff)/template_sizes[1])
    d_misid = np.sqrt(misid*(1-misid)/template_sizes[0])

    d_frac = np.sqrt((d_misid*(1-fraction))**2
                     + (d_eff*fraction)**2)/(eff-misid)

    return d_frac
