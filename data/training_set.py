
import time
import ROOT
# import argparse
import numpy as np
import matplotlib.pyplot as plt
from import_functions import loadmass_uproot, distr_extraction, train_arr_setup, array_generator


def training_set(Num=1000, f0=0.5, outputfile="prova_trset.txt"):
    """
    """
    file_mc_pi = 'root_files/tree_B0PiPi_mc.root'
    file_mc_k = 'root_files/tree_B0sKK_mc.root'
    #tree1 = "t_M0pipi;1"
    tree = 't_M0pipi;2'
    var = 'M0_Mpipi'

    arr_mc_pi, arr_mc_k = loadmass_uproot(file_mc_pi, file_mc_k, tree, var)

    c1 = ROOT.TCanvas()
    c1.cd()
    # this is fine but the optimal way would be to create a function which returns a TH1 object
    histo_mc_pi = ROOT.TH1D(
        'histo_mc_pi', 'Mpipi distribution for Pion MC', 500, 5., 5.5)
    for event in arr_mc_pi:
        histo_mc_pi.Fill(event)
    histo_mc_pi.Draw()
    c1.SaveAs("Mpipi_pi_MC.png")

    c2 = ROOT.TCanvas()
    c2.cd()
    # We use TH1 instead of arrays because the former have the GetRandom Method
    histo_mc_k = ROOT.TH1D(
        "histo_mc_pi", "Mpipi distribution for Kaon MC", 500, 5., 5.5)
    for event in arr_mc_k:
        histo_mc_k.Fill(event)
    histo_mc_k.Draw()
    c2.SaveAs("Mpipi_K_MC.png")

    # ev_pi = distr_extraction(h_pi, int((1.-f)*N), 0)
    # ev_k = distr_extraction(h_k, int(f*N), 1)
    # arr = train_arr_setup(ev_pi, ev_k)

    arr_trainingset = array_generator(
        histo_mc_pi, histo_mc_k, f0_train, N_train)
    np.savetxt("0.4_nparray.txt", arr_trainingset)
    return arr_trainingset


if __name__ == "__main__":
    t1 = time.time()
    plt.figure(1)
    N_train = 100000  # Total number of events in the training sample
    f0_train = 0.4  # Pure Fraction of K events in the training sample
    outputfile = "0.4_nparray.txt"
    arr_trainingset = training_set(N_train, f0_train, outputfile)
    plt.hist(arr_trainingset[:, 0], bins=100, histtype='step')
    plt.show()
    t2 = time.time()
    print(f'Tempo impiegato = {t2-t1} s')
