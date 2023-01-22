"""
Module that performs an esteem of the fraction f of kaons in the mixed pi-k
population present in the dataset Bhh_data.root
"""

import time
import ROOT
import numpy as np


def fit_on_templates(df, var, pars, Nbins=100,
                     h_name="h", h_title="h", LowLim=5., UpLim=5.5):
    """
    """
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1111)
    h = df.Histo1D((h_name, h_title, Nbins, LowLim, UpLim), var)
    f0 = ROOT.TF1("f0", "gaus", LowLim, UpLim)

    [f0.SetParameter(k, pars[k]) for k in range(len(pars))]
    fitvar = h.Fit("f0", "SQL")
    '''
    c = ROOT.TCanvas()
    c.cd()
    h.Draw()
    f0.Draw("same")
    c.SaveAs("prova2.png")
    '''
    print(f'Chi2 = {fitvar.Chi2}')
    print(f'Ndof = {fitvar.Ndf}')
    print(f'Prob = {fitvar.Prob}')

    exit_pars = (fitvar.GetParameter(0), fitvar.GetParameter(1),
                 fitvar.GetParameter(2))

    return exit_pars


def template_composition(model1, model2):
    """
    """


def fit_on_data(df, var, par_pi, par_k, model_pi, model_k, Nbins=100,
                h_name="h", h_title="h", LowLim=5., UpLim=5.5):
    """
    """
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1111)
    h = df.Histo1D((h_name, h_title, Nbins, LowLim, UpLim), var)

    f_pi = ROOT.TF1("f_pi", model_pi, LowLim, UpLim)
    f_k = ROOT.TF1("f_k", model_k, LowLim, UpLim)
    f_data = ROOT.TF1("f_data", "(1-[0])*f_pi + [0]*f_k", LowLim, UpLim)


if __name__ == "__main__":
    file1 = '../root_files/tree_B0PiPi_mc.root'
    file2 = '../root_files/tree_B0sKK_mc.root'
    tree = 't_M0pipi;2'
    var = 'M0_Mpipi'

    df_pi, df_k = ROOT.RDataFrame(tree, file1), ROOT.RDataFrame(tree, file2)

    pars = (10000, 5.28, 0.02)

    exit_pars = fit_on_templates(df_pi, var, pars)
