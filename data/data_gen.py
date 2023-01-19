
import time
import ROOT
# import argparse
import numpy as np
import matplotlib.pyplot as plt
from import_functions import loadmass_uproot, array_generator


def data_gen(Num=10000, outputfile="prova_data.txt"):
    """
    """
    file_mc_pi = '../root_files/tree_B0PiPi_mc.root'
    file_mc_k = '../root_files/tree_B0sKK_mc.root'
    tree = 't_M0pipi;2'
    #tree1 = "t_M0pipi;1"
    var = 'M0_Mpipi'

    seed = int(time.time())
    gen = ROOT.TRandom3()
    gen.SetSeed(seed)
    f0 = gen.Uniform(0.3, 0.7)

    arr_mc_pi, arr_mc_k = loadmass_uproot(file_mc_pi, file_mc_k, tree, var)

    c1 = ROOT.TCanvas()
    c1.cd()
    # this is fine but the optimal way would be to create a function which returns a TH1 object
    histo_mc_pi = ROOT.TH1D(
        'histo_mc_pi', 'Mpipi distribution for Pion MC', 100, 5., 5.5)
    for event in arr_mc_pi:
        histo_mc_pi.Fill(event)
    histo_mc_pi.Draw()
    c1.SaveAs("Mpipi_pi_MC.png")

    c2 = ROOT.TCanvas()
    c2.cd()
    # We use TH1 instead of arrays because the former have the GetRandom Method
    histo_mc_k = ROOT.TH1D(
        "histo_mc_k", "Mpipi distribution for Kaon MC", 100, 5., 5.5)
    for event in arr_mc_k:
        histo_mc_k.Fill(event)
    histo_mc_k.Draw()
    c2.SaveAs("Mpipi_K_MC.png")


    arr_data = array_generator(histo_mc_pi, histo_mc_k, f0_train, N_train)
    np.savetxt(outputfile, arr_data)

    return arr_data


if __name__ == "__main__":
    t1 = time.time()
    plt.figure(1)
    N_train = 100000  # Total number of events in the training sample
    f0_train = 0.4  # Pure Fraction of K events in the training sample
    outputfile = "0.4_nparray.txt"
    arr_data = data_gen(N_train, f0_train, outputfile)
    plt.hist(arr_data[:, 0], bins=100, histtype='step')
    plt.show()
    t2 = time.time()
    print(f'Tempo impiegato = {t2-t1} s')
