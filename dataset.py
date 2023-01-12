
import time
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from import_functions import loadmass_uproot, distr_extraction, train_arr_setup


t1 = time.time()
mc_pi = "data/tree_B0PiPi_mc.root"
mc_k = "data/tree_B0sKK_mc.root"
tree1 = "t_M0pipi;1"
tree2 = "t_M0pipi;2"
var = "M0_Mpipi"

pi_data, k_data = loadmass_uproot(mc_pi, mc_k, tree1, var)

c1 = ROOT.TCanvas()
c1.cd()
h_pi = ROOT.TH1F("h_pi", "h_pi", 100, 5., 5.5)
for ev1 in pi_data:
    h_pi.Fill(ev1)
h_pi.Draw()
c1.SaveAs("Pi_distribution_MC.png")


c2 = ROOT.TCanvas()
c2.cd()
h_k = ROOT.TH1F("h_k", "h_k", 100, 5., 5.5)
for ev2 in k_data:
    h_k.Fill(ev2)
h_k.Draw()
c2.SaveAs("K_distribution_MC.png")

N = 1000  # Total number of events in the training sample
f = 0.5  # Fraction of K events in the training sample
ev_pi = distr_extraction(h_pi, int((1.-f)*N), 0)
ev_k = distr_extraction(h_k, int(f*N), 1)

arr = train_arr_setup(ev_pi, ev_k)
print(arr[:20])
