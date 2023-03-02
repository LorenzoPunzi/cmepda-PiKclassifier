
""" Unit tests for dataset.py module """

import time
import unittest
import ROOT
import numpy as np
from import_functions import loadmass_uproot, distr_extraction, train_arr_setup


def cpp_tree_writing():
    code = """
        TFile f1("data/testfile1.root", "RECREATE");
        TTree data1("tree", "tree");
        double x;
        data1.Branch("x", &x, "x/D");
        for (int i=0; i<4096; i++) {
            x = gRandom->Uniform(-10,10);
            data1.Fill();
            }
        data1.Write();
        f1.Close();
        """
    return code


class TestFunctions(unittest.TestCase):
    """
    Class for unit testing
    """

    def test_extraction(self):
        """ Test for the variability of the arrays generated via the
        distr_extraction function and for the correct id flag """
        ROOT.gInterpreter.ProcessLine(cpp_tree_writing())
        data_1, data_1_copy = loadmass_uproot(
            "data/testfile1.root", "data/testfile1.root", "tree;1", "x")
        h1 = ROOT.TH1F("h1", "h1", 50, -10, 10)
        for ev in data_1:
            h1.Fill(ev)
        N_ev = 1000
        v1 = distr_extraction(h1, N_ev, 0)
        v2 = distr_extraction(h1, N_ev, 1)
        for i in range(0, N_ev):
            self.assertNotAlmostEqual(v1[i, 0], v2[i, 0])
            self.assertNotEqual(v1[i, 1], v2[i, 1])


    # def test_mixing(self):
        """ Test for the correct mixing of arrays in the train_arr_setup
        function """


if __name__ == "__main__":
    unittest.main()
