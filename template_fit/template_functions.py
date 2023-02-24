"""
"""

import ROOT
import os
import numpy as np


def InitializeFunctionsLibrary():
    """
    Function that loads into the Python environment the shared library
    containing the fit functions.
    """
    load_head = ROOT.gInterpreter.Load('#include "fit_functions.h"')
    if not load_head:
        print("ERROR in header load")
        quit()
    success = ROOT.gSystem.CompileMacro('fit_functions.cpp', opt="ks")
    success = False
    if not success:
        lib_file = ''
        for root, dirnames, filenames in os.walk("."):
            for filename in filenames:
                if '.so' in filename:
                    lib_file = os.path.join(root, filename)
        if lib_file == '':
            print("ERROR in source code compilation")
            quit()
        else:
            ROOT.gSystem.Load(lib_file)


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
                   pars=(1.4e5, 0.6, 5.28, 0.04, 5.29, 0.02)):
    """
    Returns a TF1 function corresponding to a weighted sum of two Gaussians;
    its expression is written in a C++ shared library and is imported in the
    Python environment by the InitializeFunctionsLibrary() function.

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
    InitializeFunctionsLibrary()

    function = ROOT.TF1(funcname, ROOT.DoubleGaussian, LowLim, UpLim, 6)

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
    Gaussian; its expression is written in a C++ shared library and is imported
    in the Python environment by the InitializeFunctionsLibrary() function.

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
    InitializeFunctionsLibrary()

    function = ROOT.TF1(funcname, ROOT.GaussJohnson, LowLim, UpLim, 8)
    [function.SetParameter(k, pars[k]) for k in range(8)]

    function.SetParLimits(1, 0, 1)
    [function.SetParLimits(k, 0., 10.) for k in range(2, 6)]
    function.SetParLimits(6, LowLim, UpLim)
    function.SetParLimits(7, 0, UpLim-LowLim)

    return function


if __name__ == "__main__":

    c = ROOT.TCanvas()
    c.cd()

    a = GaussJohnson(5., 5.6, pars=(
                     1.4e05, 0.99, 1.75, 0.043, 5.29, 0.85, 5.19, 0.06))

    a.Draw()
    c.SaveAs("prova_func.png")
