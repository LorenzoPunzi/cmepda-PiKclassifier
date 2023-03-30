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
    Generates mixed signal+background datasets to be analysed, starting from root files of
    toyMC events of background only and sognal only processes.

    :param filepaths_in: 2 element tuple with path to the toyMC, first being backgorund, the second being signal species.
    :type filepaths_in: list[str] or tuple[str]
    :param filepaths_out: Three element tuple of .root file paths. The first should indicate the root file containing the "background" species (flag=0), the second the "signal" species (flag=1), the third the mix to be generated.
    :type filepaths_out: list[str] or tuple[str]
    :param tree: Name of the tree in which the desired variables are stored on the toyMC files
    :type tree: str
    :param num_mc: Number of events generated for each output MC file. If both are set to zero!!!!
    :type num_mc: int
    :param num_data: Number of events generated for the output mixed file
    :type num_data: int
    :param fraction: Ideal fraction of signal events in the generated mixed sample. Actual fraction will be different if fraction*num_data is not an integer.
    :type fraction: double
    :param vars: List or tuple of variables to export from the toyMC files
    :type vars: list[str] or tuple[str]
    
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
    #      the following conditions:  n_evts_toymc_pi == num_mc+num_pions,
    #                                 n_evts_toymc_k == num_mc+num_kaons
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
            if (num_pions+num_mc > n_evts_toymc_pi) or \
               (num_kaons+num_mc > n_evts_toymc_k):
                raise IncorrectNumGenError(num_mc, num_pions+num_kaons, n_evts_toymc_pi, n_evts_toymc_k)
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

    np.random.shuffle(var_array)

    var_dictionary = {}
    for idx in range(len(vars)):
        var_dictionary.update({vars[idx]: var_array[:, idx]})

    file = uproot.recreate(filepaths_out[2])
    file[tree] = var_dictionary
    file[tree].show()
    file[tree].close()


if __name__ == '__main__':
    t1 = time.time()
    gen_from_toy(fraction=0.42, num_mc=0, num_data=0)
    t2 = time.time()
    print(f'Time elapsed = {t2-t1} s')
