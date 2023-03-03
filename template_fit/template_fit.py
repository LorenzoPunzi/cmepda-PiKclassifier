"""
Module that performs an estimate of the fraction f of kaons in the mixed pi-k
population present in the dataset Bhh_data.root
"""

import ROOT
import numpy as np
from template_functions import DoubleGaussian, GaussJohnson
from utilities.utils import default_rootpaths


def fit_on_montecarlo(df, var, fitfunc, img_name="template.png", Nbins=1000,
                      h_name="M0_MPiPi", h_title="M_pipi distribution (MC)",
                      savefig=False):
    """
    """
    ROOT.gStyle.SetOptFit(11111)
    h = df.Histo1D((h_name, h_title, Nbins, 5.0, 5.6), var)

    if savefig is True:
        fitvar = h.Fit(fitfunc, "SQLR")
        c = ROOT.TCanvas()
        c.cd()
        h.Draw()
        fitfunc.Draw("same")
        c.SaveAs(img_name)
    else:
        fitvar = h.Fit(fitfunc, "SQLRN")

    print(" ")
    print("MC template - fit results:")
    print(f'  Chi2 = {fitvar.Chi2()}')
    print(f'  Ndof = {fitvar.Ndf()}')
    print(f'  Prob = {fitvar.Prob()}')

    exit_pars = []
    for k in range(fitfunc.GetNpar()):
        exit_pars.append(fitfunc.GetParameter(k))

    return exit_pars


def fit_templates(filepaths, var, img_name="fig/Fit_data.png",
                  Nbins=1000, LowLim=5.02, UpLim=5.42,
                  h_name="M0_MPiPi", h_title="M_pipi distribution (data)",
                  savefigs=[False, False, False]):
    """
    """
    df_mc1, df_mc2, df_data = [ROOT.RDataFrame(tree, filepath) for filepath in filepaths]

    templ_pars_pi = fit_on_montecarlo(df_mc1, var, DoubleGaussian(5.02, 5.42),
                                      img_name='fig/template_fit_pi.png',
                                      h_title='M_pipi distribution (B0->PiPi MC)',
                                      savefig=savefigs[0])
    templ_pars_k = fit_on_montecarlo(df_mc2, var, GaussJohnson(5.02, 5.42),
                                     img_name='fig/template_fit_k.png',
                                     h_title='M_pipi distribution (B0s->KK MC)',
                                     savefig=savefigs[1])
    # ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(11111)
    h = df_data.Histo1D((h_name, h_title, Nbins, 5.0, 5.6), var)

    f_data = ROOT.TF1(
              "f_data", ROOT.TemplateComposition, LowLim, UpLim, 16)
    # Per tenere sotto controllo che fa TemplateComposition è meglio costruire
    # la rispettiva funzuione in model_functions.py

    [f_data.FixParameter(2+k, templ_pars_k[k]) for k in range(8)]
    [f_data.FixParameter(10+k, templ_pars_pi[k]) for k in range(6)]

    if savefigs[2] is True:
        fitvar = h.Fit(f_data, "SQLR")
        c = ROOT.TCanvas()
        c.cd()
        h.Draw()
        f_data.Draw("same")
        c.SaveAs(img_name)
    else:
        fitvar = h.Fit(f_data, "SQLRN")

    print(" ")
    print("Data - fit results:")
    print(f'  Chi2 = {fitvar.Chi2()}')
    print(f'  Ndof = {fitvar.Ndf()}')
    print(f'  Prob = {fitvar.Prob()}')

    return fitvar


if __name__ == "__main__":

    filepaths = default_rootpaths()
    tree = 'tree;1'
    var = 'M0_Mpipi'

    results = fit_templates(filepaths, var)

    print(" ")
    print(
        f'Frazione di K = {results.Parameters()[1]} +- {results.Errors()[1]}')
