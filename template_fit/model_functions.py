"""
"""

import ROOT
import numpy as np


def BreitWigner(LowLim, UpLim, funcname='BW', pars=(2000, 5.28, 0.5)):
    """
    Returns a TF1 function corresponding to a relativistic Breit-Wigner.
    The parameters are written in C++ ROOT style and represent the following
    quantities:
        [0] = Normalization
        [1] = Mass
        [2] = Gamma

    Parameters
    ----------
    LowLim: double
        Lower limit of the x axis
    UpLim: double
        Upper limit of the x axis
    funcname: string
        Name of the function created
    pars: tuple of double
        Initial parameters given to the function
    """
    arg_gamma = 'TMath::Sqrt([1]*[1]*([2]*[2] + [1]*[1]))'
    arg_kappa = '([1]*[2]*' + arg_gamma + \
        '/TMath::Sqrt([1]*[1] + ' + arg_gamma + '))'
    arg_denom_1 = '(x*x - [1]*[1])*(x*x - [1]*[1])'
    str = '[0]*6e-4*0.9*' + arg_kappa + \
        '/(' + arg_denom_1 + '+ [1]*[1]*[2]*[2])'

    function = ROOT.TF1(funcname, str, LowLim, UpLim, 3)
    print(type(function))

    [function.SetParameter(k, pars[k]) for k in range(3)]
    function.SetParLimits(0, 0., 1e10)
    function.SetParLimits(1, LowLim, UpLim)
    function.SetParLimits(2, 0., UpLim-LowLim)

    return function


def DoubleGaussian(LowLim, UpLim, funcname='DoubleGaussian',
                   pars=(1.4e5, 0.5, 5.28, 0.02, 5.28, 0.02)):
    """
    Returns a TF1 function corresponding to a weighted sum of two Gaussians.
    The parameters are written in C++ ROOT style and represent the following
    quantities:
        [0] = Normalization
        [1] = Fraction of first gaussian elements in distribution
        [2], [3] = Mean and Sigma for first Gaussian
        [4], [5] = Mean and Sigma for second Gaussian

    Parameters
    ----------
    LowLim: double
        Lower limit of the x axis
    UpLim: double
        Upper limit of the x axis
    funcname: string
        Name of the function created
    pars: tuple of double
        Initial parameters given to the function
    """
    const = '1/2.5066'
    gaus1 = 'TMath::Exp(-(x-[2])*(x-[2])/(2*[3]))/[3]'
    gaus2 = 'TMath::Exp(-(x-[4])*(x-[4])/(2*[5]))/[5]'
    str = '[0]*6e-4*([1]*' + gaus1 + ' + (1-[1])*' + gaus2 + ')*' + const

    function = ROOT.TF1(funcname, str, LowLim, UpLim, 6)

    [function.SetParameter(k, pars[k]) for k in range(6)]

    function.SetParLimits(1, 0, 1)
    function.SetParLimits(2, LowLim, UpLim)
    function.SetParLimits(3, 0, UpLim-LowLim)
    function.SetParLimits(4, LowLim, UpLim)
    function.SetParLimits(5, 0, UpLim-LowLim)

    return function


def GaussJohnson(LowLim, UpLim, funcname='Gauss+Johnson',
                 pars=(1.4e05, 0.99, 1.75, 0.043, 5.29, 0.85, 5.19, 0.06)):
    """
    Returns a TF1 function corresponding to a weighted sum of a Johnson and a
    Gaussian. The parameters in the definition are written in C++ ROOT style
    and represent the following quantities:
        [0] = Normalization
        [1] = Fraction of first gaussian elements in distribution
        [2], [3], [4], [5] = Johnson parameters (need to be positive!)
        [6], [7] = Mean and Sigma for Gaussian

    Parameters
    ----------
    LowLim: double
        Lower limit of the x axis
    UpLim: double
        Upper limit of the x axis
    funcname: string
        Name of the function created
    pars: tuple of double
        Initial parameters given to the function
    """
    const = '1/2.5066'
    arg = '((x-[4])/[3])'
    expo_arg = '([5]+[2]*TMath::ASinH' + arg + ')'
    expo = 'TMath::Exp(-0.5*' + expo_arg + '*' + expo_arg + ')'
    denom = 'TMath::Sqrt(1+(' + arg + '*' + arg + '))'
    johns = '([2]/[3])*' + expo + '/' + denom
    gaus = 'TMath::Exp(-(x-[6])*(x-[6])/(2*[7]))/[7]'
    str = '[0]*6e-4*([1]*' + johns + ' + (1-[1])*' + gaus + ')*' + const

    function = ROOT.TF1(funcname, str, LowLim, UpLim, 8)
    [function.SetParameter(k, pars[k]) for k in range(8)]

    function.SetParLimits(1, 0, 1)
    [function.SetParLimits(k, 0., 10.) for k in range(2, 6)]
    function.SetParLimits(6, LowLim, UpLim)
    function.SetParLimits(7, 0, UpLim-LowLim)

    return function

    """
def func_dictionary():
    dictionary = {}

    dictionary["BreitWigner"] = BreitWigner()
    dictionary["DoubleGaussian"] = DoubleGaussian()
    dictionary["GaussJohnson"] = GaussJohnson()

    return dictionary
"""


if __name__ == "__main__":

    c = ROOT.TCanvas()
    c.cd()

    a = BreitWigner(5, 5.6, pars=(1, 5.28, 0.02))

    # d = func_dictionary()
    # print(d.values())

    a.Draw()
    c.SaveAs("prova_func.png")
