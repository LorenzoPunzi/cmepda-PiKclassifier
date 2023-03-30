"""
"""

import unittest
import os
import matplotlib.pyplot as plt
from var_cut.var_cut import var_cut


current_dir = os.path.dirname(__file__)

# uniformly distributed between 0 and 1.0
path_A = os.path.join(current_dir, 'dummy/dummy_A.root')
# uniformly distributed between 0.5 and 1.5
path_B = os.path.join(current_dir, 'dummy/dummy_B.root')


class TestVarCut(unittest.TestCase):
    def test_uniform(self):
        """
        Test var_cut() with a two uniformly distributed histograms
        """
        rootpaths = (path_A, path_B, path_A)
        eff = 0.9

        fr, stats, _, = var_cut(rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar',
                                eff=eff, error_optimization=False,
                                inverse_mode=False, specificity_mode=False, savefig=False)
        efficiency = stats[0]
        misid = stats[1]
        cut = stats[2]

        self.assertGreaterEqual(efficiency, misid)
        self.assertAlmostEqual(fr[0], 0, delta=1e-3)
        self.assertAlmostEqual(efficiency, eff, delta=1e-3)
        self.assertAlmostEqual(misid, 0.4, delta=1e-3)
        self.assertAlmostEqual(cut, 0.6, delta=1e-3)

    def test_uniform_inverse(self):
        """
        Test var_cut() with a two uniformly distributed histograms in inverse mode
        """
        rootpaths = (path_B, path_A, path_B)
        eff = 0.9

        fr, stats, _, = var_cut(rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar',
                                eff=eff, error_optimization=False,
                                inverse_mode=True, specificity_mode=False, savefig=False)
        efficiency = stats[0]
        misid = stats[1]
        cut = stats[2]

        self.assertGreaterEqual(efficiency, misid)
        self.assertAlmostEqual(fr[0], 0, delta=1e-3)
        self.assertAlmostEqual(efficiency, eff, delta=1e-3)
        self.assertAlmostEqual(misid, 0.4, delta=1e-3)
        self.assertAlmostEqual(cut, 0.9, delta=1e-3)

    def test_uniform_specificity(self):
        """
        Test var_cut() with a two uniformly distributed histograms in specificity mode
        """
        rootpaths = (path_A, path_B, path_A)
        eff = 0.9

        fr, stats, _, = var_cut(rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar',
                                eff=eff, error_optimization=False,
                                inverse_mode=False, specificity_mode=True, savefig=False)
        efficiency = stats[0]
        misid = stats[1]
        cut = stats[2]

        self.assertGreaterEqual(efficiency, misid)
        self.assertAlmostEqual(fr[0], 0, delta=1e-3)
        self.assertAlmostEqual(efficiency, 0.6, delta=1e-3)
        self.assertAlmostEqual(misid, 1-eff, delta=1e-3)
        self.assertAlmostEqual(cut, 0.9, delta=1e-3)

    def test_uniform_inverse_specificity(self):
        """
        Test var_cut() with a two uniformly distributed histograms in inverse and specificity mode
        """
        rootpaths = (path_B, path_A, path_B)
        eff = 0.9

        fr, stats, _, = var_cut(rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar',
                                eff=eff, error_optimization=False,
                                inverse_mode=True, specificity_mode=True, savefig=False)
        efficiency = stats[0]
        misid = stats[1]
        cut = stats[2]

        self.assertGreaterEqual(efficiency, misid)
        self.assertAlmostEqual(fr[0], 0, delta=1e-3)
        self.assertAlmostEqual(efficiency, 0.6, delta=1e-3)
        self.assertAlmostEqual(misid, 1-eff, delta=1e-3)
        self.assertAlmostEqual(cut, 0.6, delta=1e-3)

    def test_uniform_false_inverse(self):
        """
        Test var_cut() with a two uniformly distributed histograms when inverse mode is given but should not be
        """
        rootpaths = (path_A, path_B, path_A)
        eff = 0.9

        fr, stats, _, = var_cut(rootpaths=rootpaths, tree='dummytree', cut_var='dummyvar',
                                eff=eff, error_optimization=False,
                                inverse_mode=True, specificity_mode=False, savefig=False)
        efficiency = stats[0]
        misid = stats[1]
        cut = stats[2]
        self.assertGreaterEqual(efficiency, misid)
        self.assertAlmostEqual(fr[0], 0, delta=1e-3)
        self.assertAlmostEqual(efficiency, eff, delta=1e-3)
        self.assertAlmostEqual(misid, 0.4, delta=1e-3)
        self.assertAlmostEqual(cut, 0.6, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
    plt.show()
