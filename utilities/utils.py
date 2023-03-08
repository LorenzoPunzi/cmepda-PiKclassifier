import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics


def default_rootpaths():
    """
    """
    current_path = os.path.dirname(__file__)
    rel_path = '../data/root_files'
    rootnames = ['B0PiPi_MC.root', 'B0sKK_MC.root', 'Bhh_data.root']
    rootpaths = tuple([os.path.join(current_path, rel_path, file)
                       for file in rootnames])
    return rootpaths


def default_txtpaths():
    """
    """
    current_path = os.path.dirname(__file__)
    rel_path = '../data/txt'
    txtnames = ['train_array.txt', 'data_array.txt']
    txtpaths = tuple([os.path.join(current_path, rel_path, file)
                      for file in txtnames])
    return txtpaths


def default_vars():
    """
    """
    return ('M0_Mpipi', 'M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_p', 'M0_pt',
            'M0_eta', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2',
            'h2_thetaC0', 'h2_thetaC1', 'h2_thetaC2')


def find_cut(pi_array, k_array, efficiency,
             specificity_mode=False, inverse_mode=False):
    """
    """
    if inverse_mode:
        efficiency = 1-efficiency
    cut = -np.sort(-k_array)[int(efficiency*(len(k_array)-1))
                             ] if not specificity_mode else np.sort(pi_array)[int(efficiency*(len(k_array)-1))]
    if inverse_mode:  # !!!! NOT DRY
        misid = (pi_array < cut).sum(
        )/pi_array.size if not specificity_mode else (k_array < cut).sum()/k_array.size
    else:
        misid = (pi_array > cut).sum(
        )/pi_array.size if not specificity_mode else (k_array > cut).sum()/k_array.size

    # .sum() sums how many True there are in the masked (xx_array>y)

    return cut, misid


def plot_rocs(rocx_array, rocy_array, roc_labels, roc_linestyles, roc_colors,
              x_pnts=(), y_pnts=(), point_labels=(''), eff=0,
              figtitle='ROC', figname='roc'):
    """
    """
    plt.figure(figtitle)
    plt.xlabel('False Positive Probability')
    plt.xlim(0, 1)
    plt.ylabel('True Positive Probability')
    plt.ylim(0, 1)
    for idx in range(len(rocx_array)):
        '''
        flag_inverse = inverse_mode_array[idx]
        if flag_inverse is True:
            rocx_array[idx] = np.ones(rocx_array[idx].size) - rocx_array[idx]
            rocy_array[idx] = np.ones(rocy_array[idx].size) - rocy_array[idx]
            auc_array[idx] = 1 - auc_array[idx]
        '''
        plt.plot(rocx_array[idx], rocy_array[idx], label=roc_labels[idx],
                 color=roc_colors[idx], linestyle=roc_linestyles[idx])
    for idx in range(len(x_pnts)):
        plt.plot((x_pnts[idx]), (y_pnts[idx]),
                 label=point_labels[idx], marker='o')

    if eff != 0:
        plt.axhline(y=eff, color='green', linestyle='--',
                    label='Efficiency chosen at ' + str(eff))
    plt.axline((0, 0), (1, 1), linestyle='--', label='AUC = 0.5')
    plt.legend()
    plt.draw()
    plt.savefig('fig/' + figname + '.pdf')

    # !!! How to make it so it saves in the /fig folder of the directory from which the function is CALLED, not the one where nnoutputfit.py IS.
    # figpath = os.path.join(os.path.dirname(__file__), "fig")
    # print(f"Figure {figname} saved in {figpath} folder")


def roc(pi_array, k_array, inverse_mode=False, makefig=False, eff=0, name="ROC"):
    """
    """
    true_array = np.concatenate(
        (np.zeros(pi_array.size), np.ones(k_array.size)))
    y_array = np.concatenate((pi_array, k_array))
    rocx, rocy, _ = metrics.roc_curve(true_array, y_array)
    auc = metrics.roc_auc_score(true_array, y_array)

    if inverse_mode:
        rocx, rocy = np.ones(rocx.size)-rocx, np.ones(rocy.size)-rocy
        auc = 1 - auc

    print(f'AUC of the ROC is {auc}')

    if makefig:
        plot_rocs([rocx], [rocy], [name], ["-"],
                  ["blue"], eff=eff, figname=name)
        '''
        plt.figure('ROC')
        plt.plot(rocx, rocy, label='ROC curve', color='red')
        plt.xlabel('False Positive Probability')
        plt.xlim(0, 1)
        plt.ylabel('True Positive Probability')
        plt.ylim(0, 1)
        plt.draw()

        if eff:
            plt.axhline(y=eff_line, color='green', linestyle='--', label='efficiency chosen at '+str(eff_line)
                        )
        plt.axline((0, 0), (1, 1), linestyle='--', label='AUC = 0.5')
        plt.legend()

        # !!! How to make it so it saves in the /fig folder of the directory from which the function is CALLED, not the one where nnoutputfit.py IS.
        current_path = os.path.dirname(__file__)
        rel_path = 'fig'
        figurepath = os.path.join(current_path, rel_path, 'roc.pdf')
        plt.savefig(figurepath)
        '''

        return rocx, rocy, auc
