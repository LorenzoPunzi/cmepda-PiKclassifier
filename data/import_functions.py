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

""" This is the straightforward approach to loading data, UPROOT is preferred (see below)
def loadmass(file, tree):
    f = ROOT.TFile.Open(file, "READ")
    t = f.Get(tree)
    n_events = t.GetEntries()
    v1 = np.ones(n_events)*999
    k = 0
    for evt in t:
        v1[k] = evt.M0_Mpipi
        k += 1
    print(v1)
    return v1
"""


# not used, in the classic template fit the RDF scheme is preferred
def loadmass_uproot(file_pi, file_k, tree, var):
    """
    Returns numpy arrays containing the variables requested, stored originally
    in root files

    Parameters
    ----------
    file_pi: string
        MC file of B0->PiPi process
    file_k: string
        MC file of B0s->KK process
    tree: string
        Tree containing the dataset (is the same for both the files)
    var: string
        Name of the branch containing the variables selected
    """
    """
    if (len(vars) == 0): # DOES THIS MAKE SENSE? OR DO WE NOT NEED IT IF THE FUNCTION IS CALLED INTERNALLY?
        pass
    tree_pi , tree_k = uproot.open(file_pi)[tree] , uproot.open(file_k)[tree]
    for variable in vars:
        var_pi , var_k = t_pi[variable].array(
            library='np') , t_k[variable].array(library='np')
    """
    t_pi, t_k = uproot.open(file_pi)[tree], uproot.open(file_k)[tree]
    v_pi, v_k = t_pi[var].array(library='np'), t_k[var].array(library='np')
    return v_pi, v_k


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

    if (len(vars) == 0):  # DOES THIS MAKE SENSE? OR DO WE NOT NEED IT IF THE FUNCTION IS CALLED INTERNALLY?
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


def distr_extraction(histo, num, flag):  # not used
    """
    Returns a numpy array of random numbers extracted from the MC distribution
    and assigns a flag number to identify the particle type

    Parameters
    ----------
    histo: ROOT.TH1
        Histogram of the MC distribution of the variable considered
    num: int
        Number of extractions requested
    flag: int (0,1)
        Id. number of the species: 0 for Pi, 1 for K
    """
    seed = int(time.time())
    gen = ROOT.TRandom3()
    gen.SetSeed(seed)
    arr = np.ones((num, 2))
    for k in range(0, num):
        arr[k, 0] = histo.GetRandom()
        arr[k, 1] = flag
    return arr


def array_generator(histo_pi, histo_k, f0, n_entries):  # not used
    """
    """
    seed = int(time.time())  # THIS IS NOT DRY!!!!!!! maybe gRandom but WHERE??
    gen = ROOT.TRandom3()
    gen.SetSeed(seed)
    arr = np.ones((n_entries, 2))
    # Perhaps it can be done more neatly with for elem in arr or something like that
    for k in range(0, n_entries):
        flag = gen.Binomial(1, f0)  # Bernoulli
        arr[k, 1] = flag
        if (flag):  # NOT DRY
            arr[k, 0] = histo_k.GetRandom()
        else:
            arr[k, 0] = histo_pi.GetRandom()

    return arr


def train_arr_setup(pi_events, k_events):  # not used
    """
    Function that concatenates and merges two arrays. If the arrays have more
    than one dimension, the operations are executed on the first dimension (
    i.e. on the rows of a 2D matrix)

    Parameters
    ----------
    pi_events: numpy array
        Array of a variable of interest belonging to simulated B0->PiPi events
    k_events: numpy array
        Array of a variable of interest belonging to simulated B0s->KK events
    """
    tr_array = np.concatenate((pi_events, k_events), axis=0)
    np.random.shuffle(tr_array)
    return tr_array


def array_to_txt(arr, filename):
    """Simple function that saves an array in a .txt file into the /txt folder
    """
    current_path = os.path.dirname(__file__)
    path = os.path.join(current_path, filename)
    np.savetxt(path, arr)


def merge_variables(*var):
    """
    Function that merges some variables with an empirically established
    function. Returns the arrays of the new defined variables
    """


if __name__ == '__main__':
    t1 = time.time()

    seed = int(time.time())  # THIS IS NOT DRY!!!!!!! maybe gRandom but WHERE??
    gen = ROOT.TRandom3()
    gen.SetSeed(seed)

    file1 = '../root_files/tree_B0PiPi.root'
    file2 = '../root_files/tree_B0sKK.root'
    tree = 't_M0pipi;1'
    var = ('M0_Mpipi', 'M0_MKK', 'M0_MKpi',
           'M0_MpiK', 'M0_p', 'M0_pt', 'M0_eta', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2')

    #v = loadmass(file, tree)
    # v1, v2 = loadmass_uproot(file1, file2, tree, var)
    v1, v2 = loadvars(file1, file2, tree, *var)
    h = ROOT.TH1F('h', 'h', 50, 5., 5.5)
    for ev in v1[:, 1]:
        h.Fill(ev)

    train_array = np.concatenate((v1[:10000, :], v2[:10000, :]), axis=0)
    np.random.shuffle(train_array)
    print(train_array.shape)

    np.savetxt("txt/train_array_prova.txt", train_array)

    tree = 't_M0pipi;1'
    file_data = '../root_files/tree_Bhh_data.root'
    v0, v0 = loadvars(file_data, file_data, tree, *var)
    np.savetxt("txt/data_array_prova.txt", v0[:1000, :])

    c1 = ROOT.TCanvas()
    h.DrawCopy()
    c1.SaveAs("prova.png")

    t2 = time.time()

    '''
    a = np.array([0, 1, 2, 3, 4])
    array_to_txt(a, "txt/dummyarray.txt")
    '''

    print(f'Tempo totale = {t2-t1}')
