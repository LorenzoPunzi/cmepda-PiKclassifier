"""
Module containing functions that import the dataset from the TTrees and store
the variables of interest in numpy arrays for ML treatment
"""

import sys
import warnings
import uproot
import corner
import numpy as np
import matplotlib.pyplot as plt
from utilities.merge_variables import mergevar
from utilities.utils import default_rootpaths, default_vars, default_figpath
from utilities.exceptions import IncorrectIterableError

warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'


def loadvars(file_pi, file_k, tree, vars, flag_column=False, flatten1d=True):
    """
    Function that extracts the chosen variables for all eventsin two
    ROOT files given and stores them in numpy arrays.

    :param file_pi: Path to MC root file of background only processes.
    :type file_pi: str
    :param file_k: Path to MC root file of signal only processes.
    :type file_k: str
    :param tree: Tree in which the variables are stored on the root files.
    :type tree: str
    :param vars: List or tuple containing names of the variables to be loaded.
    :type vars: list[str] or tuple[str]
    :param flag_column: If is ``True``, a column full of 0 or 1, for background or signal events respectively, is appended as the last column of the 2D array.
    :type flag_column: bool
    :param flatten1d: If is ``True`` and only one variable is passed as "vars", the arrays generated are returned as row-arrays instead of one-column arrays.
    :type flatten1d: bool
    :return: Two 2D numpy arrays filled by events of the two root files given in input and containing the requested variables, plus a flag-column if requested.
    :rtype: 2D numpy.array[float]
    """

    tree_pi, tree_k = uproot.open(file_pi)[tree], uproot.open(file_k)[tree]

    flag_pi = np.zeros(tree_pi.num_entries)
    flag_k = np.ones(tree_k.num_entries)

    list_pi, list_k = [], []

    for variable in vars:
        list_pi.append(tree_pi[variable].array(library='np'))
        list_k.append(tree_k[variable].array(library='np'))

    if flag_column:
        list_pi.append(flag_pi)
        list_k.append(flag_k)

    v_pi, v_k = np.stack(list_pi, axis=1), np.stack(list_k, axis=1)

    if (len(vars) == 1 and flatten1d is True):
        # Otherwise becomes a 1D column vector when 1D
        v_pi, v_k = v_pi.flatten(), v_k.flatten()

    # Shuffling of the rows in MC sets
    np.random.shuffle(v_pi)
    np.random.shuffle(v_k)

    return v_pi, v_k


def overlaid_cornerplot(rootpaths=default_rootpaths(), tree='t_M0pipi;1',
                        vars=default_vars(), figpath=''):
    """
    Generates and saves cornerplots for two different (multidimensional)
    arrays on the same canvas.

    :param figpaths: Two element list or tuple containing the two root file paths where the events are stored.
    :type figpaths: list[str] or tuple[str]
    :param tree: Name of the tree where the events are stored
    :type tree: str
    :param vars: List or tuple of names of the variables to plot.
    :type vars: list[str] or tuple[str]
    :param figpath: Path where the figure is saved. This string must not contain the name of the figure itself since it is given automatically.
    :type figpath: str

    """
    if len(rootpaths) >= 3:
        msg = '***WARNING*** \nFilepaths given are more than three. \
Using only the first three...\n*************\n'
        warnings.warn(msg, stacklevel=2)
        rootpaths = rootpaths[:3]
    try:
        if len(rootpaths) < 2 or not (type(rootpaths) == list or type(rootpaths) == tuple):
            raise IncorrectIterableError(rootpaths, 2, 'rootpaths')
    except IncorrectIterableError as err:
        print(err)
        sys.exit()

    arr_mc_pi, arr_mc_k = loadvars(rootpaths[0], rootpaths[1],
                                   tree, vars, flag_column=False)

    # To exclude RICH events that have not triggered (=999)
    mask_pi = (arr_mc_pi < 999).all(axis=1)
    mask_k = (arr_mc_k < 999).all(axis=1)

    array_tuple = (arr_mc_pi[mask_pi, :], arr_mc_k[mask_k, :])
    num = len(array_tuple)

    colormap = plt.cm.get_cmap('gist_rainbow', num+10)
    colors = [colormap(i) for i in range(num+10)]
    figure = corner.corner(
        array_tuple[0], bins=100, color=colors[0], labels=vars)

    figure.suptitle("Corner-plot of some variables")
    for i in range(1, num):
        figure = corner.corner(array_tuple[i], bins=100, fig=figure,
                               color=colors[i+7], labels=vars)

    figure.set_size_inches(9, 7)

    plt.savefig(default_figpath('corner_'+'_'.join(vars))) if figpath == '' \
        else plt.savefig(figpath+'/corner_'+'_'.join(vars)+'.pdf')


def include_merged_variables(rootpaths, tree, initial_vars, new_variables):
    """
    Function that allows to append to the existing datasets (numpy arrays) new
    columns filled by the outputs of the mergevar() function.

    :param rootpaths: Three element list or tuple of .root file paths. The first should indicate the root file containing the "background" species (flag=0), the second the "signal" species (flag=1), the third the data mix.
    :type rootpaths: list[str] or tuple[str]
    :param tree: Tree in which the variables are stored on the root files.
    :type tree: str
    :param initial_vars: List or tuple containing names of the variables to be loaded.
    :type initial_vars: list[str] or tuple[str]
    :param new_variables: List or tuple containing two element lists or tuples of variables to merge.
    :type new_variables: list[tuple[str]] or tuple[tuple[str]]
    :return: A list or tuple containing the new numpy arrays for the three datasets, with the new columns filled with the data retrieved by the merge-variables algorithm. For MC datasets the flag column is still the rightmost column.
    :rtype: list[2D numpy.array[double]] or tuple[2D numpy.array[double]]

    """

    if len(rootpaths) >= 4:
        msg = f'***WARNING*** \nInput rootpaths given are more than three. Using only the first three...\n*************\n'
        warnings.warn(msg, stacklevel=2)
    try:
        if len(rootpaths) < 3 or not (type(rootpaths) is list or type(rootpaths) is tuple):
            raise IncorrectIterableError(rootpaths, 3, 'rootpaths')
    except IncorrectIterableError as err:
        print(err)
        sys.exit()

    for var in new_variables:
        try:
            if len(var) != 2 or not (type(var) is list or type(var) is tuple):
                raise IncorrectIterableError(var, 2, 'new_variables')
        except IncorrectIterableError as err:
            print(err)
            sys.exit()

    v1, v2 = loadvars(rootpaths[0], rootpaths[1], tree, vars)
    v_data, _ = loadvars(rootpaths[2], rootpaths[2],
                         tree, vars, flag_column=False)

    n_old_vars = len(v_data[0, :])
    new_arrays = []

    l0_pi = len(v1[:, 0])
    l0_k = len(v2[:, 0])
    if l0_pi != l0_k:
        length = min(l0_pi, l0_k)
    else:
        length = l0_pi

    list_pi, list_k, list_data = [], [], []

    for newvars in new_variables:
        merged_arrays, _, _ = mergevar(rootpaths, tree, newvars,
                                       savefig=False, savetxt=False)
        list_pi.append(merged_arrays[0][:length])
        list_k.append(merged_arrays[1][:length])
        list_data.append(merged_arrays[2])

    for col in range(n_old_vars):
        list_pi.append(v1[:length, col])
        list_k.append(v2[:length, col])
        list_data.append(v_data[:, col])

    # Appending the flag column to mc arrays
    list_pi.append(v1[:length, -1])
    list_k.append(v2[:length, -1])

    new_arrays = (np.stack(list_pi, axis=1), np.stack(list_k, axis=1),
                  np.stack(list_data, axis=1))

    return new_arrays


def array_generator(rootpaths, tree, vars, n_mc=560000, n_data=50000,
                    for_training=True, for_testing=True, new_variables=()):
    """
    Generates arrays for ML treatment (training and testing). To guarantee
    unbiasedness the training array has an equal number of background and signal
    events.

    :param rootpaths: Three element list or tuple of .root file paths. The first should indicate the root file containing the "background" species (flag=0), the second the "signal" species (flag=1), the third the mix.
    :type rootpaths: list[str] or tuple[str]
    :param tree: Tree in which the variables are stored on the root files.
    :type tree: str
    :param vars: Tuple containing names of the variables to be used.
    :type vars: list[str] or tuple[str]
    :param n_mc: Number of events to take from the root files for the training set.
    :type n_mc: int
    :param n_data: Number of events to take from the root file for the testing set.
    :type n_data: int
    :param for_training: If True generates scambled dataset for training from the mc sets with flags.
    :type for_training: bool
    :param for_testing: If True generates dataset to be evaluated, hence without flag column.
    :type for_testing: bool
    :param new_variables: Optional list or tuple containing two element lists or tuples of variables to merge.
    :type new_variables: list[tuple[str]] or tuple[tuple[str]]
    :return: Two element tuple containing 2D numpy arrays. The first contains the MC datasets' events (scrambled to avoid position bias) and the flag that identifies each event as background or signal. The second contains the events of the mixed dataset without flags (one less column).
    :rtype: list[2D numpy.array[double]] or tuple[2D numpy.array[double]]
    """

    if len(rootpaths) >= 4:
        msg = f'***WARNING*** \nInput filepaths given are more than three.\
Using only the first two...\n*************\n'
        warnings.warn(msg, stacklevel=2)
        rootpaths = rootpaths[:3]
    try:
        if len(rootpaths) < 3 or not (type(rootpaths) is list or type(rootpaths) is tuple):
            raise IncorrectIterableError(rootpaths, 3, 'rootpaths')
    except IncorrectIterableError as err:
        print(err)
        sys.exit()

    train_array, data_array = np.zeros(len(vars)), np.zeros(len(vars))

    if len(new_variables) == 0:
        v_mc_pi, v_mc_k = loadvars(
            rootpaths[0], rootpaths[1], tree, vars, flag_column=True)
        train_array = np.concatenate((v_mc_pi[:int(n_mc/2), :],
                                      v_mc_k[:int(n_mc/2), :]), axis=0)
        np.random.shuffle(train_array)

        v_data, _ = loadvars(
            rootpaths[2], rootpaths[2], tree, vars, flag_column=False)
        data_array = v_data[:n_data, :]

    # If a mixing is requested, both the training and the testing arrays are
    # modified, with obviously the same mixing
    else:
        [v_mc_pi, v_mc_k, v_data_new] = include_merged_variables(
            rootpaths, tree, vars, new_variables)
        train_array = np.concatenate(
            (v_mc_pi[:int(n_mc/2), :], v_mc_k[:int(n_mc/2), :]), axis=0)
        data_array = v_data_new[:n_data, :]

    return train_array, data_array


if __name__ == '__main__':
    print('Running this module as main module is not supported. Feel free to \
add a custom main or run the package as a whole (see README.md)')
