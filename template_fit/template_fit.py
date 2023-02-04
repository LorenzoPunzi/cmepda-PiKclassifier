"""
Module that performs an esteem of the fraction f of kaons in the mixed pi-k
population present in the dataset Bhh_data.root
"""

import time
import ROOT
import numpy as np
from template_functions import DoubleGaussian, GaussJohnson


def fit_on_montecarlo(df, var, fitfunc, img_name="template.png", Nbins=1000,
                      h_name="h", h_title="h", LowLim=5.0, UpLim=5.6):
    """
    """
    ROOT.gStyle.SetOptFit(11111)
    h = df.Histo1D((h_name, h_title, Nbins, LowLim, UpLim), var)

    # [fitfunc.SetParameter(k, pars[k]) for k in range(len(pars))]
    fitvar = h.Fit(fitfunc, "SQLR")
    '''
    c = ROOT.TCanvas()
    c.cd()
    h.Draw()
    fitfunc.Draw("same")
    c.SaveAs(img_name)
    '''
    print(f'Chi2 = {fitvar.Chi2()}')
    print(f'Ndof = {fitvar.Ndf()}')
    print(f'Prob = {fitvar.Prob()}')

    exit_pars = []
    for k in range(fitfunc.GetNpar()):
        exit_pars.append(fitfunc.GetParameter(k))

    return exit_pars


def fit_on_data(df, var, templ_pars_pi, templ_pars_k, img_name="Fit_data.png",
                Nbins=1000, h_name="h", h_title="h", LowLim=5., UpLim=5.6):
    """
    """
    #ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(11111)
    h = df.Histo1D((h_name, h_title, Nbins, LowLim, UpLim), var)

    f_data = ROOT.TF1("f_data", ROOT.TemplateComposition, LowLim, UpLim, 16)
    # Per tenere sotto controllo che fa TemplateComposition è meglio costruire
    # la rispettiva funzuione in model_functions.py

    [f_data.FixParameter(2+k, templ_pars_k[k]) for k in range(8)]
    [f_data.FixParameter(10+k, templ_pars_pi[k]) for k in range(6)]

    fitvar = h.Fit(f_data, "SQLR")

    c = ROOT.TCanvas()
    c.cd()
    h.Draw()

    f_data.Draw("same")
    c.SaveAs(img_name)

    print(
        f'Frazione di K = {f_data.GetParameter(1)} +- {f_data.GetParError(1)}')
    print(f'Chi2 = {fitvar.Chi2()}')
    print(f'Ndof = {fitvar.Ndf()}')
    print(f'Prob = {fitvar.Prob()}')


if __name__ == "__main__":
    file1 = '../root_files/tree_B0PiPi_mc.root'
    file2 = '../root_files/tree_B0sKK_mc.root'
    file_data = '../root_files/tree_Bhh_data.root'
    tree = 't_M0pipi;2'
    var = 'M0_Mpipi'

    df_pi, df_k = ROOT.RDataFrame(tree, file1), ROOT.RDataFrame(tree, file2)

    templ_pi = fit_on_montecarlo(df_pi, var, DoubleGaussian(5.02, 5.42),
                                 img_name='fig/template_fit_pi.png',
                                 h_title='M_pipi distribution (B0->PiPi MC)')
    templ_k = fit_on_montecarlo(df_k, var, GaussJohnson(5.02, 5.42),
                                img_name='fig/template_fit_k.png',
                                h_title='M_pipi distribution (B0s->KK MC)')

    df_data = ROOT.RDataFrame('t_M0pipi;1', file_data)

    fit_on_data(df_data, var, templ_pi, templ_k, LowLim=5.02, UpLim=5.42)
