"""
Generates overlapping corner plots for the specified numpy arrays in order to
show which features have different distribution in the two cases
"""

import corner
from matplotlib import pyplot as plt
from utilities.import_datasets import loadvars
from utilities.utils import default_rootpaths, default_figpath, default_vars


def cornerplot(array, figname):
    """
    Generates and saves a cornerplot for a single, multi-dimensional, array

    :param array: Multidimensional array containing the values of the variables, for each event (rows) and for each feature (columns)
    :type array: 2D numpy.array[float]
    :param figname: Name of the figure that the function saves
    :type figname: str

    """
    array_new = array[:, :-1]
    figure = corner.corner(array_new)
    figure.set_size_inches(9, 7)
    figure.savefig(figname)


def overlaid_cornerplot(filepaths=default_rootpaths(), tree='t_M0pipi;1',
                        vars=default_vars(), figpath=''):
    """
    Generates and saves cornerplots for two different (multidimensional)
    arrays on the same canva. The datasets are passed as numpy ndarray and the
    corner() function plot them as histograms, for each column (= feature).

    :param figpaths: Tree paths where the events are stored
    :type figpaths: tuple[str]
    :param tree: Name of the tree where the events are stored
    :type tree: str
    :param vars: List containing the names of the variables to plot
    :type vars: tuple[str]
    :param figpath: Path where the figure is saved. This string must not contain a name for the figure since it is given automatically.
    :type figpath: str

    """

    arr_mc_pi, arr_mc_k = loadvars(filepaths[0], filepaths[1],
                                   tree, vars, flag_column=False)

    mask_pi = (arr_mc_pi < 999).all(axis=1)
    mask_k = (arr_mc_k < 999).all(axis=1)

    array_tuple = (arr_mc_pi[mask_pi, :], arr_mc_k[mask_k, :])
    num = len(array_tuple)

    colormap = plt.cm.get_cmap('gist_rainbow', num+10)
    colors = [colormap(i) for i in range(num+10)]
    figure = corner.corner(
        array_tuple[0], bins=100, color=colors[0], labels=vars)

    figure.suptitle("Corner-plot of some variables")
    for i in range(1, num):
        figure = corner.corner(array_tuple[i], bins=100, fig=figure,
                               color=colors[i+7], labels=vars)

    figure.set_size_inches(9, 7)

    plt.savefig(default_figpath('corner_'+'_'.join(vars))) if figpath == '' \
        else plt.savefig(figpath+'/corner_'+'_'.join(vars)+'.pdf')


if __name__ == '__main__':

    file_mc_pi, file_mc_k, _ = default_rootpaths()
    tree = 't_M0pipi;1'
    vars = default_vars()
    print(vars)

    variables = vars[:4]
    # variables = vars[7:]
    # variables = ['M0_p', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2']
    # variables = ['M0_p', 'h2_thetaC0', 'h2_thetaC1', 'h2_thetaC2']

    overlaid_cornerplot((file_mc_pi, file_mc_k), tree, variables)
    plt.show()
