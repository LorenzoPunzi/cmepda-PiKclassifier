"""
Module containing functions that import the dataset from the TTrees and store
the variables of interest in a numpy array for ML treatment
"""

import time
import ROOT
import uproot
import numpy as np
import matplotlib.pyplot as plt


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


def loadmass_uproot(file_pi, file_k, tree1, var):
    t1_pi = uproot.open(file_pi)[tree1]
    # t2_pi = uproot.open(file_pi)[tree2]
    v_pi = t1_pi[var].array(library="np")  # + t2_pi[var].array(library="np")
    t1_k = uproot.open(file_k)[tree1]
    # t2_k = uproot.open(file_k)[tree2]
    v_k = t1_k[var].array(library="np")  # + t2_k[var].array(library="np")
    return v_pi, v_k
    # Vedere se si può scrivere in maniera più compatta, ad es. con liste
    # Si possono usare entrambi i tree in ogni file?


def distr_extraction(histo, num, flag):
    gen = ROOT.TRandom()
    arr = np.ones((num, 2))
    for k in range(0, num):
        arr[k][0] = histo.GetRandom(gen)  # sistemare la questione del seed
        arr[k][1] = flag
    return arr


def train_arr_setup(pi_events, k_events):
    tr_array = np.concatenate((pi_events, k_events), axis=0)
    np.random.shuffle(tr_array)
    return tr_array


if __name__ == "__main__":
    t1 = time.time()
    file1 = "data/tree_B0PiPi_mc.root"
    file2 = "data/tree_B0PiPi_mc.root"
    tree1 = "t_M0pipi;1"
    tree2 = "t_M0pipi;2"
    var = "M0_Mpipi"
    #v = loadmass(file, tree)
    v1, v2 = loadmass_uproot(file1, file2, tree2, "M0_Mpipi")
    h = ROOT.TH1F("h", "h", 50, 5., 5.5)
    for ev in v1:
        h.Fill(ev)
    ex = distr_extraction(h, 5, 1)
    print(ex)

    x = train_arr_setup(ex, ex)
    print(x)

    c1 = ROOT.TCanvas()
    h.DrawCopy()
    # c1.SaveAs("prova.png")def sei_bellissimo():

    t2 = time.time()

    print("Tempo totale = %f" % (t2-t1))
