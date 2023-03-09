"""
Generates overlapping corner plots between the specified numpy arrays in order
to show which features have different distribution in the two cases
"""

import sys
import corner
from matplotlib import pyplot as plt
from utilities.import_datasets import loadvars
from utilities.utils import default_figpath



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
    plt.savefig(default_figpath(figname))


if __name__ == '__main__':

    file_mc_pi = '../root_files/tree_B0PiPi.root'
    file_mc_k = '../root_files/tree_B0sKK.root'
    tree = 't_M0pipi;1'
    if len(sys.argv) == 1:
        print('No variables given to display!')
        sys.exit(0)
    vars = [var for var in sys.argv[1:]]

    arr_mc_pi, arr_mc_k = loadvars(file_mc_pi, file_mc_k, tree, *vars)
    mask_pi = (arr_mc_pi<999).all(axis=1)
    mask_k = (arr_mc_k<999).all(axis=1)
    overlaid_cornerplot([arr_mc_pi[mask_pi,:], arr_mc_k[mask_k,:]], ["pions", "kaons"],
                        'corner_'+'_'.join(vars), vars)
    plt.show()
