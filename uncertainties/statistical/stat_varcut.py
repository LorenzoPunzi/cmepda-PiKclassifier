"""
"""
import numpy as np
from matplotlib import pyplot as plt
from var_cut.var_cut import var_cut


if __name__ == '__main__':

    var_cut(eff=0.95, inverse_mode=True, specificity_mode=False, draw_roc=True, stat_split = 100)

    plt.show()
