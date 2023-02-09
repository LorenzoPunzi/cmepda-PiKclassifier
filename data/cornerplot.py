"""
Generates overlapping corner plots between the specified numpy arrays in order
to show which features have different distribution in the two cases
"""

import numpy as np
import corner
from matplotlib import pyplot as plt
from import_functions import loadvars


def cornerplot(array, figname):
    """
    """
    array_new = array[:, :-1]
    figure = corner.corner(array_new)
    figure.set_size_inches(9, 7)
    figure.savefig(figname)
    #corner.overplot_lines(figure, np.mean(array_new,axis=0), color="C1")
    #corner.overplot_lines(figure, np.mean(array_new,axis=1), color="C2")


def overlaid_cornerplot(array_list, array_labels, figname, variable_names):
    """
    """
    num = len(array_list)
    colormap = plt.cm.get_cmap('gist_rainbow', num+10)
    colors = [colormap(i) for i in range(num+10)]
    figure = corner.corner(
        array_list[0][:, :-1], bins=100, color=colors[0], labels=variable_names)

    for i in range(1, num):
        figure = corner.corner(
            array_list[i][:, :-1], bins=100, fig=figure, color=colors[i+7], labels=variable_names)

    figure.set_size_inches(9, 7)
    plt.savefig(figname)


if __name__ == '___main__':

    file_mc_pi = '../root_files/tree_B0PiPi_mc.root'
    file_mc_k = '../root_files/tree_B0sKK_mc.root'
    tree = 't_M0pipi;2'
    vars = ('M0_Mpipi', 'M0_MKK', 'M0_p', 'M0_eta')
    # 'h1_p', 'h1_eta', 'h2_p', 'h2_eta')
    # vars = ('h1_IP', 'M0_px', 'M0_py', 'M0_pz', 'M0_pt')

    arr_mc_pi, arr_mc_k = loadvars(file_mc_pi, file_mc_k, tree, *vars)

#cornerplot(arr_mc_pi, 'fig/cornerplot_pi.png')
#cornerplot(arr_mc_k, 'fig/cornerplot_k.png')
    overlaid_cornerplot([arr_mc_pi, arr_mc_k], ["pions", "kaons"],
                        'fig/overlaid_cornerplot_'+'_'.join(vars)+'.pdf', vars)
    plt.show()
