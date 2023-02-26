import unittest
import sys
import time
import ROOT
import numpy as np
from template_fit.template_fit import fit_templates
# from templatefit.template_functions import DoubleGaussian, GaussJohnson


file1 = '../root_files/tree_B0PiPi.root'
file2 = '../root_files/tree_B0sKK.root'
file_data = '../root_files/tree_Bhh_data.root'
tree = 't_M0pipi;1'
var = 'M0_Mpipi'

df_pi, df_k = ROOT.RDataFrame(tree, file1), ROOT.RDataFrame(tree, file2)


class TestTemplateFit(unittest.TestCase):
    def test_onlypions(self):
        """
        """
        print('Running only_pions test')
        res = fit_templates([df_pi, df_k, df_pi], var, LowLim=5.02, UpLim=5.42)
        estim_f = round(res.Parameters()[1], 2)
        self.assertAlmostEqual(estim_f, 0)

    def test_onlykaons(self):
        """
        """
        print('Running only_kaons test')
        res = fit_templates([df_pi, df_k, df_k], var, LowLim=5.02, UpLim=5.42)
        estim_f = round(res.Parameters()[1], 2)
        self.assertAlmostEqual(estim_f, 1)


if __name__ == '__main__':
    unittest.main()
