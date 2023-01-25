"""
Module that performs an esteem of the fraction f of kaons in the mixed pi-k
population present in the dataset Bhh_data.root
"""

import time
import ROOT
import numpy as np
from model_functions import BreitWigner, DoubleGaussian, GaussJohnson


def fit_on_montecarlo(df, var, fitfunc, img_name="template.png", Nbins=1000,
                      h_name="h", h_title="h", LowLim=5.0, UpLim=5.6):
    """
    """
    ROOT.gStyle.SetOptFit(11111)
    h = df.Histo1D((h_name, h_title, Nbins, LowLim, UpLim), var)

    # [fitfunc.SetParameter(k, pars[k]) for k in range(len(pars))]
    fitvar = h.Fit(fitfunc, "SQLR")

    c = ROOT.TCanvas()
    c.cd()
    h.Draw()
    fitfunc.Draw("same")
    c.SaveAs(img_name)

    print(f'Chi2 = {fitvar.Chi2()}')
    print(f'Ndof = {fitvar.Ndf()}')
    print(f'Prob = {fitvar.Prob()}')

    '''
    exit_pars = []
    for k in range(fitfunc.GetNpar()):
        exit_pars.append(fitfunc.GetParameter(k))
    '''

    return fitfunc, h


def fit_on_data(df, var, templ_func_pi, templ_func_k, img_name="template.png",
                Nbins=1000, h_name="h", h_title="h", LowLim=5., UpLim=5.6):
    """
    """
    #ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(11111)
    h = df.Histo1D((h_name, h_title, Nbins, LowLim, UpLim), var)

    templ_func_pi.SetTitle("f_pi")
    templ_func_k.SetTitle("f_k")
    f_data = ROOT.TF1("f_data", "[0]*( 1-[1])*f_pi + [1]*f_k )", LowLim, UpLim)


if __name__ == "__main__":
    file1 = '../root_files/tree_B0PiPi_mc.root'
    file2 = '../root_files/tree_B0sKK_mc.root'
    tree = 't_M0pipi;2'
    var = 'M0_Mpipi'

    # functions = func_dictionary()

    df_pi, df_k = ROOT.RDataFrame(tree, file1), ROOT.RDataFrame(tree, file2)

    #pars = (8000, 0.5, 5.28, 0.015, 5.28, 0.015)
    #pars = (2290, 5.28, 0.016)

    templ_pi = fit_on_montecarlo(df_pi, var, DoubleGaussian(5.02, 5.42),
                                 img_name='template_fit_pi.png',
                                 h_title='M_pipi distribution (B0->PiPi MC)')
    templ_k = fit_on_montecarlo(df_k, var, GaussJohnson(5.02, 5.42),
                                img_name='template_fit_k.png',
                                h_title='M_pipi distribution (B0s->KK MC)')
