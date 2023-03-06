"""
Tests some functions inside nnoutputfit.py
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
from utilities.utils import find_cut, roc
import os


class TestFindCut(unittest.TestCase):
    def test_linspace(self):
        """
        Test find_cut() with a linearly spaced numpy array
        """
        test_array_a = np.linspace(0,0.7,1000)
        test_array_b = np.linspace(0.3,1,1000)
        for eff in np.linspace(0,1,20):
            y_cut , misid = find_cut(test_array_a,test_array_b,eff,inverse_mode= False)
            self.assertAlmostEqual((test_array_b>y_cut).sum()/test_array_b.size,eff, delta=5e-3)
            self.assertAlmostEqual((test_array_a>y_cut).sum()/test_array_a.size,misid, delta=5e-3)
    
    def test_linspace_specificity(self):
        """
        Test find_cut() with a linearly spaced numpy array in specificity mode
        """
        test_array_a = np.linspace(0,0.7,1000) # !!!! Not DRY
        test_array_b = np.linspace(0.3,1,1000)
        for eff in np.linspace(0,1,20):
            y_cut , misid = find_cut(test_array_a,test_array_b,eff,specificity_mode= True)
            self.assertAlmostEqual((test_array_a<y_cut).sum()/test_array_a.size,eff, delta=5e-3)
            self.assertAlmostEqual((test_array_b>y_cut).sum()/test_array_b.size,misid, delta=5e-3)

    def test_linspace_inverse(self):
        """
        Test find_cut() with a linearly spaced numpy array in inverse mode
        """
        test_array_a = np.linspace(0,0.7,1000) # !!!! Not DRY
        test_array_b = np.linspace(0.3,1,1000)
        for eff in np.linspace(0,1,20):
            y_cut , misid = find_cut(test_array_a,test_array_b,eff,inverse_mode= True)
            self.assertAlmostEqual((test_array_b<y_cut).sum()/test_array_b.size,eff, delta=5e-3)
            self.assertAlmostEqual((test_array_a<y_cut).sum()/test_array_a.size,misid, delta=5e-3)

    def test_linspace_inverse_specificity(self):
        """
        Test find_cut() with a linearly spaced numpy array in inverse and specificity mode
        """
        test_array_a = np.linspace(0,0.7,1000) # !!!! Not DRY
        test_array_b = np.linspace(0.3,1,1000)
        for eff in np.linspace(0,1,20):
            y_cut , misid = find_cut(test_array_a,test_array_b,eff,inverse_mode= True,specificity_mode=True)
            self.assertAlmostEqual((test_array_a>y_cut).sum()/test_array_a.size,eff, delta=5e-3)
            self.assertAlmostEqual((test_array_b<y_cut).sum()/test_array_a.size,misid, delta=5e-3)


class TestRoc(unittest.TestCase):
    def test_linspace(self):
        """
        Test roc() with a linearly spaced numpy array
        """
        test_array = np.linspace(0,1,1000)
        _ , _ , _ = roc(test_array,test_array)
        coords = np.linspace(0.05,0.95,100)
        plt.plot(coords,coords, '--', color = 'red')
        
        current_path = os.path.dirname(__file__)
        rel_path = './fig'
        figurepath = os.path.join(current_path, rel_path, 'test_linspace_roc.pdf')
        plt.savefig(figurepath)
        

if __name__ == '__main__':
    unittest.main()
    
    