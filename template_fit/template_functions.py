"""
Module that imports the fit functions from C++ source file
"""

import ROOT
import os
import sys
from utilities.exceptions import LoadHeadError


def initialize_library():
    """
    Function that loads into the Python environment the shared library
    containing the fit functions, via the Declare and CompileMacro methods.
    """
    current_path = os.path.dirname(__file__)
    header_path = os.path.join(current_path, 'fit_functions.h')
    try:
        load_head = ROOT.gInterpreter.Declare('#include "'+header_path+'"')
        if load_head is not True:
            raise LoadHeadError(header_path)
    except LoadHeadError as err:
        print(err)
        sys.exit()

    success = ROOT.gSystem.CompileMacro('fit_functions.cpp', opt="ks")
    if success is not True:
        lib_file = ''
        for root, dirnames, filenames in os.walk("."):
            for filename in filenames:
                if '.so' in filename:
                    lib_file = os.path.join(root, filename)
        if lib_file == '':
            print("ERROR in source code compilation")
            sys.exit()
        else:
            ROOT.gSystem.Load(lib_file)


def DoubleGaussian(func_limits, funcname='DoubleGaussian',
                   pars=(1e5, 0.16, 5.28, 0.08, 5.29, 0.04)):
    """
    Function that creates a ROOT::TF1 object consisting in a weighted sum of
    two Gaussians; its expression is written in a C++ shared library and is
    imported in the Python environment by the InitializeFunctionsLibrary()
    function.

    :param func_limits: Limits of the axis where the function is defined
    :type func_limits: tuple[float]
    :param funcname: Name of the TF1 created
    :type funcname: str
    :param pars: Initial parameters given to the function
    :type pars: tuple[float]
    :return: The corresponding ROOT function
    :rtype: ROOT::TF1
    """

    initialize_library()

    function = ROOT.TF1(funcname, ROOT.DoubleGaussian,
                        func_limits[0], func_limits[1], 6)

    [function.SetParameter(k, pars[k]) for k in range(6)]

    function.SetParLimits(1, 0, 1)
    function.SetParLimits(2, func_limits[0], func_limits[1])
    function.SetParLimits(3, 0, func_limits[1]-func_limits[0])
    function.SetParLimits(4, func_limits[0], func_limits[1])
    function.SetParLimits(5, 0, func_limits[1]-func_limits[0])

    return function


def GaussJohnson(func_limits, funcname='Gauss+Johnson',
                 pars=(1e05, 0.991, 1.57, 0.045, 5.29, 1.02, 5.28, 0.00043)):
    """
    Function that creates a ROOT::TF1 object constisting in a weighted sum of
    an SU-Johnson and a Gaussian; its expression is written in a C++ shared
    library and is imported in the Python environment by the InitializeFunctionsLibrary()
    function.

    :param func_limits: Limits of the axis where the function is defined
    :type func_limits: tuple[float]
    :param funcname: Name of the TF1 created
    :type funcname: str
    :param pars: Initial parameters given to the function
    :type pars: tuple[float]
    :return: The corresponding ROOT function
    :rtype: ROOT::TF1
    """

    initialize_library()

    function = ROOT.TF1(funcname, ROOT.GaussJohnson,
                        func_limits[0], func_limits[1], 8)

    [function.SetParameter(k, pars[k]) for k in range(8)]
    function.SetParLimits(1, 0, 1)
    [function.SetParLimits(k, 0., 10.) for k in range(2, 6)]
    function.SetParLimits(6, func_limits[0], func_limits[1])
    function.SetParLimits(7, 0, func_limits[1]-func_limits[0])

    return function


if __name__ == "__main__":
    print('Running this module as main module is not supported. Feel free to \
add a custom main or run the package as a whole (see README.md)')
