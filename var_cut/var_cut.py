"""
Estimates f using a cut on Mhh only
"""
import numpy as np
from matplotlib import pyplot as plt
from utilities.utils import default_rootpaths , find_cut , roc
from utilities.import_datasets import loadvars

def var_cut(tree = 'tree;1', cut_var = 'M0_Mpipi', efficiency = 0.95, rootpaths = default_rootpaths(), inverse_mode = False, specificity_mode = False, draw_roc = True, drawfig = True):
    """
    """

    eff = efficiency
    rootpath_pi, rootpath_k, rootpath_data = rootpaths
    M_pi, M_k = loadvars(rootpath_pi, rootpath_k, tree, [cut_var], flag_column=False)

    mcut, misid = find_cut(M_pi, M_k, eff,inverse_mode=inverse_mode, specificity_mode=specificity_mode)

    if drawfig:
        plt.figure('Cut on ' + cut_var)
        plt.hist(M_pi, bins=300, range=(5.0,5.6), histtype='step', 
                    color='red', label=cut_var + ' for Pions') # !!! (range)
        plt.hist(M_k, bins=300, range=(5.0,5.6), histtype='step',
                    color='blue', label=cut_var + ' for Kaons')
        plt.axvline(x=mcut, color='green', label=cut_var + ' cut for '
                    + str(eff)+' efficiency')
        plt.draw()
        plt.xlabel(cut_var)
        plt.ylabel('Events per ?????')  # !!! (binwidth)
        plt.legend()
        plt.savefig('./fig/mcut_'+cut_var+'.pdf')

    
    rocx, rocy, auc = roc(M_pi, M_k, eff_line = eff, inverse_mode= inverse_mode, makefig = draw_roc)
    
    print(f'mcut is {mcut} for {eff} efficiency')
    print(f'misid is {misid} for {eff} efficiency')

    M_data, _ = loadvars(rootpath_data, rootpath_data, tree, [cut_var], flag_column=False)
    f = ((M_data < mcut).sum()/M_data.size-misid)/(eff-misid)
    print(f'The estimated fraction of K events is {f}')

    return f, rocx, rocy, auc

if __name__ == '__main__':

    var_cut(efficiency = 0.95, inverse_mode = True, specificity_mode = False, draw_roc = True)

    plt.show()

