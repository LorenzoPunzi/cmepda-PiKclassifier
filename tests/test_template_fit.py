"""
Tests some functions inside template_fit/template_fit.py
"""
import unittest
import sys
import time
import ROOT
import numpy as np
from utilities.utils import default_rootpaths
from template_fit.template_functions import DoubleGaussian, GaussJohnson
from template_fit.template_fit_var import global_fit, fit_mc_template
# from templatefit.template_functions import DoubleGaussian, GaussJohnson


initial_fp = default_rootpaths()
tree = 't_M0pipi;1'
var = 'M0_Mpipi'
filepaths = [initial_fp[0], initial_fp[1], " "]
fit_range = (5.02, 5.42)

# df_pi, df_k = ROOT.RDataFrame(tree, file1), ROOT.RDataFrame(tree, file2)


class TestTemplateFit(unittest.TestCase):
    def test_onlypions(self):
        """
        """
        print('Running only_pions test')
        filepaths[2] = initial_fp[0]
        templ_pars_pi = fit_mc_template(
            filepaths[0], tree, var, DoubleGaussian(fit_range))
        templ_pars_k = fit_mc_template(
            filepaths[1], tree, var, GaussJohnson(fit_range))

        res = global_fit(filepaths[2], tree, var, pars_mc1=templ_pars_k,
                         pars_mc2=templ_pars_pi, savefigs=False)

        # res = global_fit(filepaths, tree, var)
        estim_f = res.Parameters()[1]
        self.assertAlmostEqual(estim_f, 0, delta=3*res.Errors()[1])

    def test_onlykaons(self):
        """
        """
        print('Running only_kaons test')
        filepaths[2] = initial_fp[1]
        templ_pars_pi = fit_mc_template(
            filepaths[0], tree, var, DoubleGaussian(fit_range))
        templ_pars_k = fit_mc_template(
            filepaths[1], tree, var, GaussJohnson(fit_range))
        res = global_fit(filepaths[2], tree, var, pars_mc1=templ_pars_k,
                         pars_mc2=templ_pars_pi, savefigs=False)
        estim_f = res.Parameters()[1]
        self.assertAlmostEqual(estim_f, 1, delta=3*res.Errors()[1])


if __name__ == '__main__':
    unittest.main()
