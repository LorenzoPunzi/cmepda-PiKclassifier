"""
Tests some functions inside nnoutputfit.py
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
from machine_learning.nnoutputfit import find_cut


class TestFindCut(unittest.TestCase):
    def test_linspace(self):
        """
        Test with a linearly spaced numpy array
        """
        test_array = np.linspace(0,1,1000)
        for eff in np.linspace(0,1,20):
            y_cut , _ = find_cut(test_array,test_array,eff,inverse_mode= False)
            self.assertAlmostEqual((test_array>y_cut).sum()/test_array.size,eff, delta=5e-3)
    
    def test_linspace_inverse(self):
        """
        Test with a linearly spaced numpy array in inverse mode
        """
        test_array = np.linspace(0,1,1000)
        for eff in np.linspace(0,1,20):
            y_cut , _ = find_cut(test_array,test_array,eff,inverse_mode= True)
            self.assertAlmostEqual((test_array<y_cut).sum()/test_array.size,eff, delta=5e-3)
        

if __name__ == '__main__':
    unittest.main()