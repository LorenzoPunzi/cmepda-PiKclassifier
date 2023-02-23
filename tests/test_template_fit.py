import unittest
import time
import ROOT
import numpy as np
from template_fit.template_functions import DoubleGaussian, GaussJohnson

class TestTemplateFit(unittest.TestCase):
    def test_onlypions(self):
        """
        Test find_cut() with a linearly spaced numpy array
        """
        test_array = np.linspace(0,1,1000)
        for eff in np.linspace(0,1,20):
            y_cut , _ = find_cut(test_array,test_array,eff,inverse_mode= False)
            self.assertAlmostEqual((test_array>y_cut).sum()/test_array.size,eff, delta=5e-3)
    
    def test_onlykaons(self):
        """
        Test find_cut() with a linearly spaced numpy array
        """
        test_array = np.linspace(0,1,1000)
        for eff in np.linspace(0,1,20):
            y_cut , _ = find_cut(test_array,test_array,eff,inverse_mode= False)
            self.assertAlmostEqual((test_array>y_cut).sum()/test_array.size,eff, delta=5e-3)
  
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