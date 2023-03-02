"""
Tests some functions inside import_functions.py
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from utilities.import_datasets import loadvars, include_merged_variables
import os


class TestLoadvars(unittest.TestCase):
    """
    """
    def test_array_construction(self):
        """
        Tests that the array extracted from a root file corresponds to the
        expected one. We use a dummy Root file with variables generate for this
        purpose
        """


class TestIncludeMergedVariables(unittest.TestCase):
    """
    """

    def test_append_training(self):
        """
        Tests the correct formation of a training array that includes the
        mixed-variable arrays previously saved
        """

    def test_append_testing(self):
        """
        Tests the correct formation of a testing array that includes the
        mixed-variable arrays previously saved
        """




if __name__ == '__main__':
