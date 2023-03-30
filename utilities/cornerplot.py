"""
Generates overlapping corner plots for the specified numpy arrays in order to
show which features have different distribution in the two cases
"""
import warnings
import sys
import corner
from matplotlib import pyplot as plt
from utilities.import_datasets import loadvars
from utilities.utils import default_rootpaths, default_figpath, default_vars
from utilities.exceptions import IncorrectIterableError

warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'

def overlaid_cornerplot(filepaths=default_rootpaths(), tree='t_M0pipi;1',
                        vars=default_vars(), figpath=''):
    """
    Generates and saves cornerplots for two different (multidimensional)
    arrays on the same canvas.

    :param figpaths: Two element list or tuple containing the two root file paths where the events are stored.
    :type figpaths: list[str] or tuple[str]
    :param tree: Name of the tree where the events are stored
    :type tree: str
    :param vars: List or tuple of names of the variables to plot.
    :type vars: list[str] or tuple[str]
    :param figpath: Path where the figure is saved. This string must not contain the name of the figure itself since it is given automatically.
    :type figpath: str

    """
    if len(filepaths)>=3:
        msg = f'***WARNING*** \nFilepaths given are more than two. Using only the first two...\n*************\n'
        warnings.warn(msg, stacklevel=2)
    try:
        if len(filepaths)<2 or not (type(filepaths)==list or type(filepaths)==tuple):
            raise IncorrectIterableError(filepaths,2,'filepaths') 
    except IncorrectIterableError as err:
        print(err)
        sys.exit()
    
    arr_mc_pi, arr_mc_k = loadvars(filepaths[0], filepaths[1],
                                   tree, vars, flag_column=False)

    mask_pi = (arr_mc_pi < 999).all(axis=1) # To exclude RICH events that have not triggered (=999)
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
    print('Running this module as main module is not supported. Feel free to add \
          a custom main or run the package as a whole (see README.md)')
