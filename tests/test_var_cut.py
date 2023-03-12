"""
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from var_cut.var_cut import var_cut


class TestVarCut(unittest.TestCase):
    def test_uniform(self):
        """
        Test var_cut() with a two uniformly distributed histograms
        """
        rootpaths=("dummy/dummy_A.root","dummy/dummy_B.root","dummy/dummy_A.root")
        efficiency = 0.9

        f, cut, misid, _, _, _ = var_cut( rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar',
            eff=0.90, inverse_mode=True, specificity_mode=True, draw_fig=True)
        print('**********')
        print(f'efficiency = {efficiency}')
        print(f'misid = {misid}')
        print(f'cut = {cut}')
        print(f'fract = {f}')
        plt.show()
        

        
        
if __name__ == '__main__':
    unittest.main()
    plt.show()

