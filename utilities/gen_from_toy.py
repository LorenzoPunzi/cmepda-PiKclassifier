"""
"""
import ROOT
import uproot
import numpy as np
import time
from utilities.utils import default_vars, default_rootpaths


def gen_from_toy(filepaths_in=('../data/root_files/toyMC_B0PiPi.root',
                               '../data/root_files/toyMC_B0sKK.root'),
                 filepaths_out=default_rootpaths(), tree='t_M0pipi;1',
                 num_mc=100000, num_data=30000, f=0.42, vars=default_vars()):
    """
    """
    dataframes = [ROOT.RDataFrame(tree, filepath) for filepath in filepaths_in]

    n_evts_toymc_pi = dataframes[0].Count().GetValue()
    #n_evts_toymc_pi = n_evts_toymc_pi.
    n_evts_toymc_k = dataframes[1].Count().GetValue()
    #n_evts_toymc_k = n_evts_toymc_k

    print(n_evts_toymc_pi, n_evts_toymc_k)

    num_species = [int((1-f)*num_data), int(f*num_data)]
    num_pions, num_kaons = num_species

    print(f'Actual fraction of Kaons = {num_kaons/num_data}')

    # Imporre una condizione che sia di "default" e che dipenda unicamente dal
    # numero di eventi nei tree in ingresso
    # if num_mc == 0 and num_data == 0:

    if (n_evts_toymc_pi < num_mc + num_pions) or (n_evts_toymc_k < num_mc + num_kaons):
        print("ERROREE")
        quit()

    # dataframes[0] = dataframes[0].Define("flag", "0")
    # dataframes[1] = dataframes[1].Define("flag", "1")

    # takes the first num_mc events of the input toyMC files
    df_mc_pi = dataframes[0].Range(num_mc)
    # creates a .root file with the chosen vars as branches
    df_mc_pi.Snapshot("tree", filepaths_out[0], vars)

    df_mc_k = dataframes[1].Range(num_mc)
    df_mc_k.Snapshot("tree", filepaths_out[1], vars)

    # takes the rest of the input toyMC to be used as data
    df_data_pi = dataframes[0].Range(num_mc, num_mc+num_species[0])
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
    for idx in range(len(vars)):
        var_dictionary.update({vars[idx]: var_array[:, idx]})
        # !!! Could be made more DRY since it's the same as AsNumpy() above? Why are we not using RDataFrame here as well?
        # The transformation df -> numpy array -> new ttree is necessary because
        # we need to shuffle the events and the easiest way to do that is with the
        # method np.random.shuffle.
        # Maybe we can avoid the second "for" loop by generating an array of indexes
        # (integers from 0 to n_data), shuffling it BEFORE the first loop; then we
        # can generate the dictionary and we have to find a way to apply this index
        # array to all the values in the var_dictionary
        # In general, I think that using uproot is the most straightforward way
        # to manage ttrees; there could be some doubts on the usage of .AsNumpy()
        # method, because I haven't found it in the documentation and it seems
        # less performant than other methods (https://indico.cern.ch/event/775679/contributions/3244724/attachments/1767054/2869505/RDataFrame.AsNumpy.pdf)

    file["tree"] = var_dictionary

    file["tree"].show()
    file["tree"].close()


if __name__ == '__main__':
    t1 = time.time()
    gen_from_toy()
    t2 = time.time()
    print(f'Time elapsed = {t2-t1} s')
