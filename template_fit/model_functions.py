"""
"""

import ROOT
import numpy as np


def BreitWigner():
    arg1 = '2./3.1416'
    arg2 = '([1]*[1])*([2]*[2])'  # Gamma=par[1]  M=par[2]
    arg3 = '((x*x) - ([2]*[2]))*((x*x) - ([2]*[2]))'
    arg4 = '(x*x*x*x)*(([1]*[1])/([2]*[2]))'

    str = '[0]*' + arg1 + '*' + arg2 + '/(' + arg3 + ' + ' + arg4 + ')'

    return str


def DoubleGaussian():
    """
    Returns a string that describes a function formed by a weighted sum of
    two Gaussians. The parameters are written in C++ ROOT style and represent
    the following quantities:
        [0] = Normalization
        [1] = Fraction of first gaussian elements in distribution
        [2], [3] = Mean and Sigma for first Gaussian
        [4], [5] = Mean and Sigma for second Gaussian
    """
    const = '1/2.5066'
    gaus1 = 'TMath::Exp(-(x-[2])*(x-[2])/(2*[3]))/[3]'
    gaus2 = 'TMath::Exp(-(x-[4])*(x-[4])/(2*[5]))/[5]'

    str = '[0]*([1]*' + gaus1 + ' + (1-[1])*' + gaus2 + ')*' + const

    return str


def GaussJohnson():
    """
    Returns a string that describes a function formed by a weighted sum of
    a Johnson and a Gaussian. The parameters are written in C++ ROOT style
    and represent the following quantities:
        [0] = Normalization
        [1] = Fraction of first gaussian elements in distribution
        [2], [3], [4], [5] = Johnson parameters (need to be positive!)
        [6], [7] = Mean and Sigma for Gaussian
    """
    const = '1/2.5066'
    arg = '((x-[4])/[3])'
    expo_arg = '([5]+[2]*TMath::ASinH' + arg + ')'
    expo = 'TMath::Exp(-0.5*' + expo_arg + '*' + expo_arg + ')'
    denom = 'TMath::Sqrt(1+(' + arg + '*' + arg + '))'
    johns = '([2]/[3])*' + expo + '/' + denom

    gaus = 'TMath::Exp(-(x-[6])*(x-[6])/(2*[7]))/[7]'

    str = '[0]*([1]*' + johns + ' + (1-[1])*' + gaus + ')*' + const

    return str


def func_dictionary():
    dictionary = {}

    dictionary["BreitWigner"] = BreitWigner()
    dictionary["DoubleGaussian"] = DoubleGaussian()
    dictionary["GaussJohnson"] = GaussJohnson()

    return dictionary


if __name__ == "__main__":
    f = DoubleGaussian()
    a = ROOT.TF1("a", f, 0., 20, 6)
    a.SetParameters(1, 0.3, 5, 2, 15, 3)

    d = func_dictionary()

    print(d.values())

    c = ROOT.TCanvas()
    c.cd()
    a.Draw()
    c.SaveAs("prova_func.png")
