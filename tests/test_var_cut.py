"""
"""

import unittest
import os
import matplotlib.pyplot as plt
from var_cut.var_cut import var_cut


current_dir = os.path.dirname(__file__)

path_A= os.path.join(current_dir, 'dummy/dummy_A.root')
path_B = os.path.join(current_dir, 'dummy/dummy_B.root')

class TestVarCut(unittest.TestCase):
    def test_uniform(self):
        """
        Test var_cut() with a two uniformly distributed histograms
        """
        rootpaths=(path_A,path_B,path_A)
        efficiency = 0.9

        f, misid, cut, _, = var_cut( rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar', eff=0.90, inverse_mode=False, specificity_mode=False, draw_fig=False)
        
        self.assertGreaterEqual(efficiency,misid)
        self.assertAlmostEqual(f[0],0, delta=1e-3)
        print('NORMAL')
        print(f'efficiency = {efficiency}')
        print(f'misid = {misid}')
        print(f'cut = {cut}')
        print(f'fract = {f[0]}')
        print('**********\n\n\n\n')

    def test_uniform_inverse(self):
        """
        Test var_cut() with a two uniformly distributed histograms in inverse mode
        """
        rootpaths=(path_B,path_A,path_B)
        efficiency = 0.9

        f, misid, cut, _, = var_cut( rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar',
            eff=0.90, inverse_mode=True, specificity_mode=False, draw_fig=False)
        
        self.assertGreaterEqual(efficiency,misid)
        self.assertAlmostEqual(f[0],0, delta=1e-3)
        print('INVERSE')
        print(f'efficiency = {efficiency}')
        print(f'misid = {misid}')
        print(f'cut = {cut}')
        print(f'fract = {f[0]}')
        print('**********\n\n\n\n')

        
    
    def test_uniform_specificity(self):
        """
        Test var_cut() with a two uniformly distributed histograms in specificity mode
        """
        rootpaths=(path_A,path_B,path_A)
        efficiency = 0.9

        f, misid, cut, _, = var_cut( rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar',
            eff=0.90, inverse_mode=False, specificity_mode=True, draw_fig=False)
        
        self.assertGreaterEqual(misid,1-efficiency)
        self.assertAlmostEqual(f[0],0, delta=1e-3)
        print('SPECIFICITY')
        print(f'efficiency = {misid}')
        print(f'misid = {1-efficiency}')
        print(f'cut = {cut}')
        print(f'fract = {f[0]}')
        print('**********\n\n\n\n')

    def test_uniform_inverse_specificity(self):
        """
        Test var_cut() with a two uniformly distributed histograms in inverse and specificity mode
        """
        rootpaths=(path_B,path_A,path_B)
        efficiency = 0.9

        f, misid, cut, _, = var_cut( rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar',
            eff=0.90, inverse_mode=True, specificity_mode=True, draw_fig=False)
        
        self.assertGreaterEqual(misid,1-efficiency)
        self.assertAlmostEqual(f[0],0, delta=1e-3)
        print('SPECIFICITY+INVERSE')
        print(f'efficiency = {misid}')
        print(f'misid = {1-efficiency}')
        print(f'cut = {cut}')
        print(f'fract = {f[0]}')
        print('**********\n\n\n\n')

    
    def test_uniform_false_inverse(self):
        """
        Test var_cut() with a two uniformly distributed histograms in inverse and specificity mode
        """
        rootpaths=(path_A,path_B,path_A)
        efficiency = 0.9

        f, misid, cut, _, = var_cut( rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar', eff=0.90, inverse_mode=True, specificity_mode=False, draw_fig=False)
        
        self.assertGreaterEqual(efficiency,misid)
        self.assertAlmostEqual(f[0],0, delta=1e-3)
        print('FALSE INVERSE')
        print(f'efficiency = {efficiency}')
        print(f'misid = {misid}')
        print(f'cut = {cut}')
        print(f'fract = {f[0]}')
        print('**********\n\n\n\n')
        

        
        
if __name__ == '__main__':
    unittest.main()
    plt.show()

