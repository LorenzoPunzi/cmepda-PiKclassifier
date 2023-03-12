"""
Generates the datasets needed for the analysis, starting from root files of
toyMC events of the processes B0->PiPi and B0s->KK.
"""
import time
import sys
import ROOT
import uproot
import numpy as np
from utilities.utils import default_vars, default_rootpaths
from utilities.exceptions import IncorrectFractionError, IncorrectNumGenError


def gen_from_toy(filepaths_in=('../data/root_files/toyMC_B0PiPi.root',
                               '../data/root_files/toyMC_B0sKK.root'),
                 filepaths_out=default_rootpaths(), tree='t_M0pipi;1',
                 num_mc=0, num_data=0, fraction=0.42, vars=default_vars()):
    """
    Generates the datasets needed for the analysis, starting from root files of
    toyMC events of the processes B0->PiPi and B0s->KK.

        Parameters:
            filepaths_in : list or tuple
                Path (with name) of the toyMC files
                Default: 'cmepda-PiKclassifier/data/root_files/toyMC_xxxx.root'
            filepaths_out : list or tuple
                Path (with name) of the three output file files
                Default: paths given by the default_rootpaths() function
            tree : string
                Name of the tree in the toyMC files where the events are stored
            num_mc : int
                Number of events in each output MC template file
                Default: 0
            num_data : int
                Number of events in the output file with mixed species
                Default: 0
            fraction : double
                Fraction of Kaon events in the mixed sample
                Default: 0.42
            vars : list or tuple
                Variables stored in the toyMC files that are stored in the
                output datasets.
                Default: variables given by the default_vars() function

        Returns:
            Two datasets that have the role of templates for the analysis and
            one file containing the required mixed fraction of events. The
            events in toyMC files are selected uniquely, to ensure indipendence
            in the three files generated.
    """

    if tree.endswith(";1"):
        tree = tree.replace(";1", "")

    try:
        if fraction >= 0.0 and fraction <= 1.0:
            pass
        else:
            raise IncorrectFractionError(fraction)
    except IncorrectFractionError as err:
        print(err)
        sys.exit()

    dataframes = [ROOT.RDataFrame(tree, filepath) for filepath in filepaths_in]

    n_evts_toymc_pi = dataframes[0].Count()
    n_evts_toymc_pi = n_evts_toymc_pi.GetValue()
    n_evts_toymc_k = dataframes[1].Count()
    n_evts_toymc_k = n_evts_toymc_k.GetValue()

    # If num_mc and num_data are BOTH set to zero, the datasets are generated
    # by taking from the toyMCs the maximum possible number of events (*) and
    # by imposing the condition num_data/(2*num_mc) = alpha
    # (*): for the cases fraction<0.5 and fraction>=0.5 respectively, we impose
    #      the followingconditions:  n_evts_toymc_pi == num_mc+num_pions,
    #                                n_evts_toymc_k == num_mc+num_kaons
    alpha = 0.2
    if int(num_mc) == 0 and int(num_data) == 0:
        if fraction < 0.5:
            num_mc = n_evts_toymc_pi/(1 + (2*alpha*(1-fraction)))
        if fraction >= 0.5:
            num_mc = n_evts_toymc_k/(1 + (2*alpha*fraction))
        num_pions, num_kaons = int(
            0.2*(1-fraction)*num_mc), int(0.2*fraction*num_mc)
        num_data = num_pions + num_kaons
        num_mc = int(num_mc)

    else:
        try:
            num_pions, num_kaons = int(
                (1-fraction)*num_data), int(fraction*num_data)
            if (num_pions+num_mc <= n_evts_toymc_pi) and \
               (num_kaons+num_mc <= n_evts_toymc_k):
                pass
            else:
                raise IncorrectNumGenError()
        except IncorrectNumGenError as err:
            print(err)
            sys.exit()

    print(f'Actual fraction of Kaons = {num_kaons/num_data}')

    # Takes the first num_mc events of the input toyMC files
    df_mc_pi = dataframes[0].Range(num_mc)
    # Creates a .root file with the chosen vars as branches
    df_mc_pi.Snapshot(tree, filepaths_out[0], vars)

    df_mc_k = dataframes[1].Range(num_mc)
    df_mc_k.Snapshot(tree, filepaths_out[1], vars)

    # Takes the rest of the input toyMC to be used as data
    df_data_pi = dataframes[0].Range(num_mc, num_mc+num_pions)
    df_data_k = dataframes[1].Range(num_mc, num_mc+num_kaons)

    var_list = []
    for var in vars:
        v_temp_pi = df_data_pi.AsNumpy()[var]
        v_temp_k = df_data_k.AsNumpy()[var]
        v_temp = np.concatenate((v_temp_pi, v_temp_k), axis=0)
        var_list.append(v_temp)

    var_array = np.stack(var_list, axis=1)
    print(np.shape(var_array))

    np.random.shuffle(var_array)

    var_dictionary = {}
    for idx in range(len(vars)):
        var_dictionary.update({vars[idx]: var_array[:, idx]})
        # !!! Could be made more DRY since it's the same as AsNumpy() above?
        # Why are we not using RDataFrame here as well?
        #
        # The transformation df -> numpy array -> new ttree is necessary
        # because we need to shuffle the events and the easiest way to do that
        # is with the method np.random.shuffle.
        # Maybe we can avoid the second "for" loop by generating an array of
        # indexes (integers from 0 to n_data), shuffling it BEFORE the first
        # loop; then we can generate the dictionary and we have to find a way
        # to apply this index array to all the values in the var_dictionary.
        # In general, I think that using uproot is the most straightforward way
        # to manage ttrees; there could be some doubts on the usage of
        # .AsNumpy() method, because I haven't found it in the documentation
        # and it seems less performant than other methods
        # (https://indico.cern.ch/event/775679/contributions/3244724/attachments/1767054/2869505/RDataFrame.AsNumpy.pdf)

    print(tree)
    file = uproot.recreate(filepaths_out[2])
    file[tree] = var_dictionary
    file[tree].show()
    file[tree].close()


if __name__ == '__main__':
    t1 = time.time()
    gen_from_toy(fraction=0.42, num_mc=0, num_data=0)
    t2 = time.time()
    print(f'Time elapsed = {t2-t1} s')
