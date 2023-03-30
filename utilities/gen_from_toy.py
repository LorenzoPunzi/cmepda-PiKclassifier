"""
Generates the datasets needed for the analysis, starting from two toy events root files
.
"""
import traceback
import sys
import warnings
import ROOT
import uproot
import numpy as np
from utilities.utils import default_vars, default_rootpaths
from utilities.exceptions import IncorrectFractionError, IncorrectNumGenError, IncorrectIterableError

warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'


def gen_from_toy(filepaths_in=('../data/root_files/toyMC_B0PiPi.root',
                               '../data/root_files/toyMC_B0sKK.root'),
                 filepaths_out=default_rootpaths(), tree='t_M0pipi;1',
                 num_mc=0, num_data=0, fraction=0.42, vars=default_vars()):
    """
    Generates mixed signal+background datasets to be analysed, starting from two root files of
    toy events of background only and signal only processes respectively.

    :param filepaths_in: 2 element tuple with path to the two toys, first being background, the second being signal species.
    :type filepaths_in: list[str] or tuple[str]
    :param filepaths_out: Three element tuple of .root file paths. The first should indicate the root file containing the "background" species (flag=0), the second the "signal" species (flag=1), the third the data mix to be generated.
    :type filepaths_out: list[str] or tuple[str]
    :param tree: Name of the tree in which the desired variables are stored in the toy files (must be the same for both files).
    :type tree: str
    :param num_mc: Number of events generated for each output MC file. If both ``num_mc`` and ``num_data` are set to zero the maximum possible number of events is extracted from the toys
    :type num_mc: int
    :param num_data: Number of events generated for the output data file (mixed). If both ``num_mc`` and ``num_data` are set to zero the maximum possible number of events is extracted from the toys
    :type num_data: int
    :param fraction: Ideal fraction of signal events in the generated mixed sample. Actual fraction will be different if fraction*num_data is not an integer.
    :type fraction: double
    :param vars: List or tuple of variables to export from the toy files.
    :type vars: list[str] or tuple[str]
    """

    if tree.endswith(";1"):
        tree = tree.replace(";1", "")

    try:
        if fraction <= 0.0 or fraction >= 1.0:
            raise IncorrectFractionError(fraction)
    except IncorrectFractionError:
        print(traceback.format_exc())
        sys.exit()

    if len(filepaths_in) >= 3:
        msg = '***WARNING*** \nInput filepaths given are more than two. \
Using only the first two...\n*************\n'
        warnings.warn(msg, stacklevel=2)
    try:
        if len(filepaths_in) < 2 or not (type(filepaths_in) == list or type(filepaths_in) == tuple):
            raise IncorrectIterableError(filepaths_in, 2, 'filepaths_in')
    except IncorrectIterableError:
        print(traceback.format_exc())
        sys.exit()

    if len(filepaths_out) >= 4:
        msg = '***WARNING*** \nOutput filepaths given are more than three. \
Using only the first three...\n*************\n'
        warnings.warn(msg, stacklevel=2)
    try:
        if len(filepaths_out) < 3 or not (type(filepaths_out) == list or type(filepaths_out) == tuple):
            raise IncorrectIterableError(filepaths_out, 3, 'filepaths_out')
    except IncorrectIterableError:
        print(traceback.format_exc())
        sys.exit()

    dataframes = [ROOT.RDataFrame(tree, filepath) for filepath in filepaths_in]

    # Number of events in the dataframes
    n_evts_toymc_pi = dataframes[0].Count()
    n_evts_toymc_pi = n_evts_toymc_pi.GetValue()
    n_evts_toymc_k = dataframes[1].Count()
    n_evts_toymc_k = n_evts_toymc_k.GetValue()

    alpha = 0.2

    # If num_mc and num_data are BOTH set to zero, the datasets are generated
    # by taking from the toyMCs the maximum possible number of events (*) and
    # by imposing the condition num_data/(2*num_mc) = alpha
    # (*): for the cases fraction<0.5 and fraction>=0.5 respectively, we impose
    #      the following conditions:  n_evts_toymc_pi == num_mc+num_pions,
    #                                 n_evts_toymc_k == num_mc+num_kaons

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
                raise IncorrectNumGenError(
                    num_mc, num_pions+num_kaons, n_evts_toymc_pi, n_evts_toymc_k)
        except IncorrectNumGenError:
            print(traceback.format_exc())
            sys.exit()

    print(f'Actual fraction of signal events = {num_kaons/num_data}')

    # Takes the first num_mc events of the input toy files
    df_mc_pi = dataframes[0].Range(num_mc)
    # Creates a .root file with the chosen vars as branches
    df_mc_pi.Snapshot(tree, filepaths_out[0], vars)

    df_mc_k = dataframes[1].Range(num_mc)
    df_mc_k.Snapshot(tree, filepaths_out[1], vars)

    # Takes the rest of the input toys to be used as data
    df_data_pi = dataframes[0].Range(num_mc, num_mc+num_pions)
    df_data_k = dataframes[1].Range(num_mc, num_mc+num_kaons)

    # Since data set needs to be shuffled, passing through numpy arrays
    var_list = []
    for var in vars:
        v_temp_pi = df_data_pi.AsNumpy()[var]
        v_temp_k = df_data_k.AsNumpy()[var]
        v_temp = np.concatenate((v_temp_pi, v_temp_k), axis=0)
        var_list.append(v_temp)

    var_array = np.stack(var_list, axis=1)

    np.random.shuffle(var_array)

    var_dictionary = {}  # Dictionary of vars to be saved in the data outfile
    for idx in range(len(vars)):
        var_dictionary.update({vars[idx]: var_array[:, idx]})

    file = uproot.recreate(filepaths_out[2])
    file[tree] = var_dictionary
    file[tree].show()
    file[tree].close()


if __name__ == '__main__':
    print('Running this module as main module is not supported. Feel free to \
add a custom main or run the package as a whole (see README.md)')
