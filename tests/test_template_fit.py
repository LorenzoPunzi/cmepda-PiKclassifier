"""
Tests some functions inside template_fit/template_fit.py
"""
import unittest
import sys
import time
"""
import ROOT
import numpy as np
from utilities.utils import default_rootpaths
from template_fit.template_functions import DoubleGaussian, GaussJohnson
from template_fit.template_fit_var import global_fit, fit_mc_template
from template_fit.template_functions import DoubleGaussian, GaussJohnson


filepaths = default_rootpaths()
tree = 't_M0pipi;1'
var = 'M0_Mpipi'
fit_range = (5.02, 5.42)

# df_pi, df_k = ROOT.RDataFrame(tree, file1), ROOT.RDataFrame(tree, file2)


class TestTemplateFit(unittest.TestCase):
    def test_onlypions(self):

        print('Running only_pions test')
        templ_pars_pi = fit_mc_template(
            filepaths[0], tree, var, DoubleGaussian(fit_range))
        templ_pars_k = fit_mc_template(
            filepaths[1], tree, var, GaussJohnson(fit_range))

        res_pi = global_fit(filepaths[0], tree, var, pars_mc1=templ_pars_k,
                            pars_mc2=templ_pars_pi, savefigs=False)
        estim_onlyPi = res_pi.Parameters()[1]
        err_onlyPi = res_pi.Errors()[1]

        self.assertAlmostEqual(estim_onlyPi, 0, places=2)

    def test_onlykaons(self):

        print('Running only_kaons test')
        templ_pars_pi = fit_mc_template(
            filepaths[0], tree, var, DoubleGaussian(fit_range))
        templ_pars_k = fit_mc_template(
            filepaths[1], tree, var, GaussJohnson(fit_range))
        res_k = global_fit(filepaths[1], tree, var, pars_mc1=templ_pars_k,
                           pars_mc2=templ_pars_pi, savefigs=False)
        estim_onlyK = res_k.Parameters()[1]
        err_onlyPi = res_k.Errors()[1]
        # print(estim_onlyK, res_k.Errors()[1])
        self.assertAlmostEqual(estim_onlyK, 1, places=2)


if __name__ == '__main__':
    unittest.main()
"""