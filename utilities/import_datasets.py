"""
Module containing functions that import the dataset from the TTrees and store
the variables of interest in numpy arrays for ML treatment
"""

import sys
import time
# import ROOT
import uproot
import numpy as np
from utilities.merge_variables import mergevar
from utilities.utils import default_rootpaths, default_vars
from utilities.exceptions import InvalidArrayGenRequestError


def loadvars(file_pi, file_k, tree, vars, flag_column=False, flatten1d=True):
    """
    Function that extracts the events of the interesting variables from the
    ROOT files and stores them in numpy arrays.

    Parameters:
        file_pi : string
            Path of MC file of B0->PiPi process
        file_k : string
            Path of MC file of B0s->KK process
        tree : string
            Tree containing the datasets
        vars : tuple
            Tuple containing names of the variables requested
            Default: variables given by the default_vars() function
        flag_column : bool
            If is set to True, a column full of 0 or 1, for Pions or Kaons
            respectively, is added at the rightmost position
            Default: False
        flatten1d : bool
            If is True and only one variable is passed as "vars", the array
            generated are returned as row-arrays
            Default: True

    Returns:
        Two numpy arrays filled by events of the two root files given in input
        and containing the requested variables, plus a flag-column if requested
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

    np.random.shuffle(v_pi)
    np.random.shuffle(v_k)

    return v_pi, v_k


def include_merged_variables(rootpaths, tree, initial_arrays, new_variables):
    """
    Function that allows to append to the existing datasets (numpy arrays) new
    columns filled by the outputs of the mergevar() function.

    Parameters:
        rootpaths : tuple
            Paths (with name) of the root files where the variables are stored.
            It must include both the MC files for Pi and K and the mixed data
            file, in this order
        tree : string
            Tree containing the datasets
        initial_arrays : tuple
            Arrays containing the data of the original variables of Pions and
            Kaons MC and of the mixed dataset
        new_variables : tuple
            Tuple containing the combination (consisting themselves in
            two-elements tuples) of variables to merge

    Returns:
        A tuple containing the new numpy arrays for the three datasets, with
        the new columns filled with the data retrieved by the merge-variables
        algorithm. For MC datasets the flag column is still the rightmost
    """

    n_old_vars = len(initial_arrays[2][0, :])
    new_arrays = []

    l0_pi = len(initial_arrays[0][:, 0])
    l0_k = len(initial_arrays[1][:, 0])
    if l0_pi != l0_k:
        length = min(l0_pi, l0_k)
    else:
        length = l0_pi

    list_pi, list_k, list_data = [], [], []

    for newvars in new_variables:
        merged_arrays, KS_stats, m = mergevar(
            rootpaths, tree, newvars, savefig=False, savetxt=False)
        list_pi.append(merged_arrays[0][:length])
        list_k.append(merged_arrays[1][:length])
        list_data.append(merged_arrays[2])

    for col in range(n_old_vars):
        list_pi.append(initial_arrays[0][:length, col])
        list_k.append(initial_arrays[1][:length, col])
        list_data.append(initial_arrays[2][:, col])

    # Appending the flag column to mc arrays
    list_pi.append(initial_arrays[0][:length, -1])
    list_k.append(initial_arrays[1][:length, -1])

    new_arrays = (np.stack(list_pi, axis=1), np.stack(list_k, axis=1),
                  np.stack(list_data, axis=1))

    return new_arrays


def array_generator(rootpaths, tree, vars, n_mc=100000, n_data=15000,
                    for_training=True, for_testing=True, new_variables=()):
    """
    Generates arrays for ML treatment (training and testing). To guarantee
    unbiasedness the training array has an equal number of Pions' and Kaons'
    events.

    Parameters
        rootpaths : tuple
            Paths of the tree files needed to generate the requested dataset(s)
        tree : string
            Tree containing the datasets
        vars : tuple
            Tuple containing names of the variables requested
            Default: variables given by the default_vars() function
        n_mc, n_data : int
            Number of total events requested for training and for testing.
        for_training, for_testing: bool
            Flags that indicate for which purpose the dataset(s) is created
            Defalut: True
        new_variables : tuple
            Tuple containing the combination (consisting themselves in
            two-elements tuples) of variables to merge
            Default: empty tuple

    Returns:
        Two numpy array: the first containing the MC datasets' events and the
        flag that identifies each event as Pi or K; the second containing the
        events of the mixed dataset. Obviously, the array for testing will have
        one column less than the other one.
    """

    try:
        if (for_training and for_testing and len(rootpaths) == 3):
            filepath_pi, filepath_k, filepath_data = rootpaths
        elif (for_training and len(rootpaths) == 2 and for_testing is not True):
            filepath_pi, filepath_k = rootpaths
        elif (for_testing and len(rootpaths) == 1 and for_training is not True):
            filepath_data = rootpaths[0]
        else:
            raise InvalidArrayGenRequestError(
                for_training, for_testing, mixed=False)
    except InvalidArrayGenRequestError as err:
        print(err)
        sys.exit()

    try:
        if len(new_variables) != 0 and len(rootpaths) == 3:
            pass
        elif len(new_variables) == 0:
            pass
        else:
            raise InvalidArrayGenRequestError(
                for_training, for_testing, mixing=True)
    except InvalidArrayGenRequestError as err:
        print(err)
        sys.exit()

    train_array, test_array = np.zeros(len(vars)), np.zeros(len(vars))

    if len(new_variables) == 0:
        if for_training:
            v_mc_pi, v_mc_k = loadvars(filepath_pi, filepath_k,
                                       tree, vars, flag_column=True)
            train_array = np.concatenate((v_mc_pi[:int(n_mc/2), :],
                                          v_mc_k[:int(n_mc/2), :]), axis=0)
            np.random.shuffle(train_array)
        if for_testing:
            v_data, _ = loadvars(filepath_data, filepath_data,
                                 tree, vars, flag_column=False)
            test_array = v_data[:n_data, :]

    # If a mixing is requested, both the training and the testing arrays are
    # modified, with obviously the same mixing
    elif len(new_variables) != 0 and len(rootpaths) == 3:
        v1, v2 = loadvars(filepath_pi, filepath_k, tree, vars)
        v_data, _ = loadvars(filepath_data, filepath_data,
                             tree, vars, flag_column=False)
        initial_arrays = [v1, v2, v_data]
        [v_mc_pi, v_mc_k, v_data_new] = include_merged_variables(
            rootpaths, tree, initial_arrays, new_variables)
        train_array = np.concatenate(
            (v_mc_pi[:int(n_mc/2), :], v_mc_k[:int(n_mc/2), :]), axis=0)
        test_array = v_data_new[:n_data, :]
    else:
        print("UNEXPECTED ERROR")
        sys.exit()

    return train_array, test_array


if __name__ == '__main__':
    t1 = time.time()

    rootpaths = default_rootpaths()

    file_pi, file_k, file_data = rootpaths

    combinations = (('M0_MKpi', 'M0_MpiK'),)

    # ('M0_MKK', 'M0_p'), ('M0_Mpipi', 'M0_p'), ('M0_MKK', 'M0_MpiK'), ('M0_Mpipi', 'M0_MKpi'))

    tree = 't_M0pipi;1'

    v_mc, v_data = array_generator(rootpaths, tree, vars=default_vars(),
                                   n_mc=560000, n_data=100000,
                                   new_variables=combinations)

    print(np.shape(v_mc), np.shape(v_data))

    # np.savetxt('../data/txt/train_array.txt', v_mc)
    # np.savetxt('../data/txt/data_array.txt', v_data)

    t2 = time.time()

    print(f'Tempo totale = {t2-t1}')
