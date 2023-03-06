"""
Tests some functions inside import_functions.py
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from utilities.import_datasets import loadvars, include_merged_variables, array_generator
import os

path_pi = 'dummy/dummy_pi.root'
path_k = 'dummy/dummy_k.root'
path_dat = 'dummy/dummy_dat.root'

class TestLoadvars(unittest.TestCase):
    """ 
    """
    def test_loadvars_flag(self):
        """
        Tests that the array extracted from a root file corresponds to the
        expected one. We use a dummy Root file with variables generate for this
        purpose
        """
        tree = 'dummytree;1'
        dummyvars = ('A','C')
        v_pi, v_k = loadvars(path_pi,path_k, tree, vars = dummyvars)

        self.assertEqual(v_pi.shape,(100,3))
        self.assertEqual(v_k.shape,(100,3))

        self.assertAlmostEqual(tuple(v_pi[:,0]),tuple(np.ones(100))) # !!!! can't figure out how to do this  with the np directly
        self.assertAlmostEqual(tuple(v_pi[:,1]),tuple(3*np.ones(100)))
        self.assertAlmostEqual(tuple(v_pi[:,2]),tuple(np.zeros(100)))

        self.assertAlmostEqual(tuple(v_k[:,0]),tuple(4*np.ones(100)))
        self.assertAlmostEqual(tuple(v_k[:,1]),tuple(6*np.ones(100)))
        self.assertAlmostEqual(tuple(v_k[:,2]),tuple(np.ones(100)))

    def test_loadvars_noflag(self):
        """
        Tests that the array extracted from a root file corresponds to the
        expected one. We use a dummy Root file with variables generate for this
        purpose
        """
        tree = 'dummytree;1'
        dummyvars = ('C','B')
        v_pi, v_k = loadvars(path_pi,path_k, tree, vars = dummyvars, flag_column=False)

        self.assertEqual(v_pi.shape,(100,2))
        self.assertEqual(v_k.shape,(100,2))

        self.assertAlmostEqual(tuple(v_pi[:,0]),tuple(3*np.ones(100)))
        self.assertAlmostEqual(tuple(v_pi[:,1]),tuple(2*np.ones(100)))

        self.assertAlmostEqual(tuple(v_k[:,0]),tuple(6*np.ones(100)))
        self.assertAlmostEqual(tuple(v_k[:,1]),tuple(5*np.ones(100)))

    def test_loadvars_1D(self):
        """
        Tests that the array extracted from a root file corresponds to the
        expected one. We use a dummy Root file with variables generate for this
        purpose
        """
        tree = 'dummytree;1'
        dummyvars = ('B')
        v_pi, v_k = loadvars(path_pi,path_k, tree, vars = dummyvars, flag_column=False, flatten1d=False)

        self.assertEqual(v_pi.shape,(100,1))
        self.assertEqual(v_k.shape,(100,1))

        self.assertAlmostEqual(tuple(v_pi[:,0]),tuple(2*np.ones(100)))
        self.assertAlmostEqual(tuple(v_k[:,0]),tuple(5*np.ones(100)))

        v_pi, v_k = loadvars(path_pi,path_k, tree, vars = dummyvars, flag_column=False, flatten1d=True)

        self.assertEqual(v_pi.shape,(100,))
        self.assertEqual(v_k.shape,(100,))

        self.assertAlmostEqual(tuple(v_pi),tuple(2*np.ones(100)))
        self.assertAlmostEqual(tuple(v_k),tuple(5*np.ones(100)))
        
        



class TestArrayGenerator(unittest.TestCase):
    """ 
    """
    def test_training_testing(self):
        """
        Tests that the array extracted from a root file corresponds to the
        expected one. We use a dummy Root file with variables generate for this
        purpose
        """
        tree = 'dummytree;1'
        dummyvars = ('A','C')
        v_mc, v_dat = array_generator((path_pi,path_k,path_dat), tree, vars = dummyvars, n_mc=100,n_data = 50)

        self.assertEqual(v_mc.shape,(100,3))
        self.assertEqual(v_dat.shape,(50,2))

        # !!!! How to assert wether the flag column is EITHER 0 or 1???
        self.assertAlmostEqual(tuple(v_dat[:,0]),tuple(7*np.ones(50)))
        self.assertAlmostEqual(tuple(v_dat[:,1]),tuple(9*np.ones(50)))


    







class TestIncludeMergedVariables(unittest.TestCase):
    """
    """

    def test_append_training(self):
        """
        Tests the correct formation of a training array that includes the
        mixed-variable arrays previously saved
        """
        pass

    def test_append_testing(self):
        """
        Tests the correct formation of a testing array that includes the
        mixed-variable arrays previously saved
        """
        pass




if __name__ == '__main__':
    unittest.main()