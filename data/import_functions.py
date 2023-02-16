"""
Module containing functions that import the dataset from the TTrees and store
the variables of interest in a numpy array for ML treatment
"""

import os
import time
import ROOT
import uproot
import numpy as np
import matplotlib.pyplot as plt


def loadvars(file_pi, file_k, tree, *vars):  # FUNDAMENTAL
    """
    Returns numpy arrays containing the requested variables, stored originally
    in MC root files. The LAST column of the output array contains a flag (0
    for Pions, 1 for Kaons)

    Parameters
    ----------
    file_pi: string
        MC file of B0->PiPi process
    file_k: string
        MC file of B0s->KK process
    tree: string
        Tree containing the dataset (is the same for both the files)
    *vars: string
        Tuple containing names of the variables requested
    """

    if (len(vars) == 0):
        # We should put an error here
        print("No variables passed to loadvars function!")
        pass

    tree_pi, tree_k = uproot.open(file_pi)[tree], uproot.open(file_k)[tree]

    flag_pi = np.zeros(tree_pi.num_entries)
    flag_k = np.ones(tree_k.num_entries)

    list_pi, list_k = [], []

    for variable in vars:
        list_pi.append(tree_pi[variable].array(library='np'))
        list_k.append(tree_k[variable].array(library='np'))

    list_pi.append(flag_pi)
    list_k.append(flag_k)

    v_pi, v_k = np.stack(list_pi, axis=1), np.stack(list_k, axis=1)

    return v_pi, v_k


def array_generator(filepaths, tree, *vars, Ntrain=50000, Ndata=10000,
                    for_training=True, for_testing=True):
    """
    """
    if for_training & for_testing:
        [filepath_pi, filepath_k, filepath_data] = filepaths
    elif len(filepaths) == 2 & for_training:
        [filepath_pi, filepath_k] = filepaths
    elif len(filepaths) == 1 & for_testing:
        filepath_data = filepaths[0]
    else:
        print('Errore nell istanziamento di array_generator()')
        pass

    if for_training:
        v1, v2 = loadvars(filepath_pi, filepath_k, tree, *vars)
        train_array = np.concatenate(
            (v1[:int(Ntrain/2), :], v2[:int(Ntrain/2), :]), axis=0)
        np.random.shuffle(train_array)
        print(train_array.shape)
        #np.savetxt("txt/training_array.txt", train_array)

    if for_testing:
        v0, v0 = loadvars(filepath_data, filepath_data, tree, *vars)
        #np.savetxt("txt/data_array_prova.txt", v0[:Ndata, :])


if __name__ == '__main__':
    t1 = time.time()

    '''
    seed = int(time.time())  # THIS IS NOT DRY!!!!!!! maybe gRandom but WHERE??
    gen = ROOT.TRandom3()
    gen.SetSeed(seed)
    '''
    current_path = os.path.dirname(__file__)
    rel_path = '../root_files'
    filenames = ['tree_B0PiPi_mc.root',
                 'tree_B0sKK_mc.root', 'tree_Bhh_data.root']

    filepaths = [os.path.join(
        current_path, rel_path, file) for file in filenames]

    print(filepaths)

    tree = 't_M0pipi;1'
    var = ('M0_Mpipi', 'M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_p',
           'M0_pt', 'M0_eta', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2')

    array_generator(filepaths, tree, *var)

    # v = loadmass(file, tree)
    # v1, v2 = loadmass_uproot(file1, file2, tree, var)

    t2 = time.time()

    print(f'Tempo totale = {t2-t1}')
