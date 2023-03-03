"""
"""
import ROOT
import uproot
import os
import numpy as np
import time 
from utilities.utils import default_vars , default_rootpaths

def gen_from_toy(filepaths_in = ('../data/root_files/toyMC_B0PiPi.root', '../data/root_files/toyMC_B0sKK.root'), filepaths_out = default_rootpaths(), tree = 't_M0pipi;1', num_mc = 250000, num_data = 50000, f = 0.42, vars = default_vars()):

    t1 = time.time()

    vars = vars+ ('flag',)
    
    dataframes = [ROOT.RDataFrame(tree, filepath) for filepath in filepaths_in]

    n_evts_toymc_pi = dataframes[0].Count().GetValue()
    n_evts_toymc_k = dataframes[1].Count().GetValue()

    num_species = [int((1-f)*num_data), int(f*num_data)]
    num_pions, num_kaons = num_species

    print(f'Actual fraction of Kaons = {num_kaons/num_data}')

    if (n_evts_toymc_pi < num_mc + num_pions) or (n_evts_toymc_k < num_mc + num_kaons):
        print("ERROREE")
    pass

    dataframes[0] = dataframes[0].Define("flag", "0")
    dataframes[1] = dataframes[1].Define("flag", "1")

    df_mc_pi = dataframes[0].Range(num_mc) # takes the first num_mc events of the input toyMC files
    df_mc_pi.Snapshot("tree", filepaths_out[0], vars) # creates a .root file with the chosen vars as branches

    df_mc_k = dataframes[1].Range(num_mc)
    df_mc_k.Snapshot("tree", filepaths_out[1], vars)

    df_data_pi = dataframes[0].Range(num_mc, num_mc+num_species[0]) # takes the rest of the input toyMC to be used as data
    # tmp_df_data_pi.Snapshot("tree_data", filenames_out[2], vars)

    df_data_k = dataframes[1].Range(num_mc, num_mc+num_species[1])
    # tmp_df_data_k.Snapshot("tree_data", filenames_out[3], vars)

    var_list = []

    file = uproot.recreate(filepaths_out[2])

    for idx in range(len(vars)):
        v_temp_pi = df_data_pi.AsNumpy()[vars[idx]]
        v_temp_k = df_data_k.AsNumpy()[vars[idx]]
        v_temp = np.concatenate((v_temp_pi, v_temp_k), axis=0)
        var_list.append(v_temp)

    var_array = np.stack(var_list, axis=1)
    print(np.shape(var_array))

    np.random.shuffle(var_array)

    var_dictionary = {}
    for idx in range(len(vars)-1):
        var_dictionary.update({vars[idx]: var_array[:, idx]}) # !!! Could be made more DRY since it's the same as AsNumpy() above? Why are we not using RDataFrame here as well?

    file["tree"] = var_dictionary

    file["tree"].show()
    file["tree"].close()

    t2 = time.time()

    print(f'Time elapsed = {t2-t1} s')
