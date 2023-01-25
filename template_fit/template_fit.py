"""
Module that performs an esteem of the fraction f of kaons in the mixed pi-k
population present in the dataset Bhh_data.root
"""

import time
import ROOT
import numpy as np
from model_functions import func_dictionary


def fit_on_montecarlo(df, var, pars, fitfunc, fitfunc_name="fitfunc", Nbins=1000,
                      h_name="h", h_title="h", LowLim=5, UpLim=5.6):
    """
    """
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1111)
    h = df.Histo1D((h_name, h_title, Nbins, LowLim, UpLim), var)

    [fitfunc.SetParameter(k, pars[k]) for k in range(len(pars))]
    fitvar = h.Fit("fitfunc", "SQLR")

    c = ROOT.TCanvas()
    c.cd()
    h.Draw()
    fitfunc.Draw("same")
    c.SaveAs("prova_template.png")

    print(f'Chi2 = {fitvar.Chi2()}')
    print(f'Ndof = {fitvar.Ndf()}')
    print(f'Prob = {fitvar.Prob()}')

    exit_pars = []
    for k in range(len(pars)):
        exit_pars.append(fitfunc.GetParameter(k))

    return exit_pars


def template_composition(model1, model2):
    """
    """


def fit_on_data(df, var, par_pi, par_k, model_pi, model_k, Nbins=1000,
                h_name="h", h_title="h", LowLim=5., UpLim=5.6):
    """
    """
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1111)
    h = df.Histo1D((h_name, h_title, Nbins, LowLim, UpLim), var)

    f_pi = ROOT.TF1("f_pi", model_pi, LowLim, UpLim)
    f_k = ROOT.TF1("f_k", model_k, LowLim, UpLim)
    f_data = ROOT.TF1("f_data", "[0]( 1-[1])*f_pi + [1]*f_k )", LowLim, UpLim)


if __name__ == "__main__":
    file1 = '../root_files/tree_B0PiPi_mc.root'
    file2 = '../root_files/tree_B0sKK_mc.root'
    tree = 't_M0pipi;2'
    var = 'M0_Mpipi'

    functions = func_dictionary()

    df_pi, df_k = ROOT.RDataFrame(tree, file1), ROOT.RDataFrame(tree, file2)

    pars = (8000, 0.5, 5.28, 0.015, 5.28, 0.015)
    #pars = (2290, 5.28, 0.016)

    fitfunc = ROOT.TF1("fitfunc", functions["DoubleGaussian"], 5.02, 5.42)
    fitfunc.SetParLimits(1, 0, 1)
    fitfunc.SetParLimits(2, 5.2, 5.3)
    fitfunc.SetParLimits(4, 5.2, 5.3)

    exit_pars = fit_on_montecarlo(df_pi, var, pars, fitfunc)
    print(exit_pars)
