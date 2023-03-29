"""
Builds and tests a Decision Tree Classifier with multiple variables (features)
in numpy arrays, performs an evaluation on a mixed dataset and applies them the
algorithm that estimates the fraction of Kaons.
"""
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from utilities.utils import default_rootpaths, default_txtpaths, default_vars,\
                            default_figpath, stat_error, syst_error
from utilities.import_datasets import array_generator
from utilities.exceptions import InvalidSourceError


def dt_classifier(source=('root', default_rootpaths()), root_tree='tree;1',
                  vars=default_vars(), n_mc=560000, n_data=50000,
                  test_size=0.3, min_leaf_samp=1, crit='gini',
                  print_tree='printed_dtc', figpath=''):
    """
    Builds and tests a Decision Tree Classifier with multiple variables (features)
    in numpy arrays, performs an evaluation on a mixed dataset and applies them the
    algorithm that estimates the fraction of Kaons.

    :param source: Two element tuple containing respectively the option for how to build the DTC and the relative paths. The first item can be either 'txt' or 'root'. In case it is built from txt the second element of source must be a tuple containing two .txt paths, one relative to the template set .txt file and the other to the set to be evaluated. The .txt files must be in a format compatible with numpy's loadtxt() and savetxt() methods. In case it is built from root, the second element of source must be a tuple containing three .root file paths: the first should indicate the root file containing the "background" species (flag=0), the second the "signal" species (flag=1), the third the mix to be evaluated.
    :type source: tuple[{'root','txt'},tuple[str]]
    :param root_tree: In case of 'root' source, the name of the tree from which to load variables.
    :type root_tree: str
    :param vars: In case of 'root' source, tuple containing the names of the variables to load and with which the DTC should be built.
    :type vars: tuple[str]
    :param n_mc: In case of 'root' source, number of events to take from the root files as template dataset
    :type n_mc: int
    :param n_data: In case of 'root' source, number of events to take from the root file as mixed dataset
    :type n_data: int
    :param test_size: The fraction of events in the template dataset to be used as testing dataset for the DTC.
    :type test_size: float
    :param min_leaf_samp: The minimum number of samples required to split an internal node. If it's an int, it's the minimum number. If it's a float, it's the fraction.
    :type min_leaf_samp: int or float
    :param crit: The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'log_loss' and 'entropy' both for the Shannon information gain.
    :type crit: {'gini','log_loss','entropy'}
    :param print_tree: If different from '', prints the tree on a .txt file with the given name
    :type print_tree: str
    :return: Estimated fraction of Kaons (with uncertainties) and parameters of the test algorithm
    :rtype: tuple[float], tuple[float]

    """
    try:
        if source[0] == 'txt':
            mc_array_path, data_array_path = source[1] if source[1] \
                else default_txtpaths()
            mc_set, data_set = np.loadtxt(mc_array_path), \
                np.loadtxt(data_array_path)
        elif source[0] == 'root':
            mc_set, data_set = array_generator(rootpaths=source[1],
                                               tree=root_tree, vars=vars,
                                               n_mc=n_mc, n_data=n_data)
        else:
            raise InvalidSourceError(source[0])
    except InvalidSourceError as err:
        print(err)
        sys.exit()

    dtc = tree.DecisionTreeClassifier(criterion=crit,
                                      min_samples_leaf=min_leaf_samp)

    X_train, X_test, y_train, y_test = train_test_split(
        mc_set[:, :-1], mc_set[:, -1], test_size=test_size, random_state=1)
    dtc = dtc.fit(X_train, y_train)

    n_nodes = dtc.tree_.node_count
    max_depth = dtc.tree_.max_depth

    print(
        f'Number of nodes of the generated decision tree classifier = {n_nodes}')
    print(f'Max depth of the generated decision tree classifier = {max_depth}')

    pi_test = np.array([X_test[i, :]
                        for i in range(np.shape(X_test)[0]) if y_test[i] == 0])
    k_test = np.array([X_test[i, :]
                       for i in range(np.shape(X_test)[0]) if y_test[i] == 1])

    pi_eval, k_eval = dtc.predict(pi_test), dtc.predict(k_test)

    data_eval = dtc.predict(data_set)

    efficiency = (k_eval == 1).sum()/k_eval.size
    misid = (pi_eval == 1).sum()/pi_eval.size
    fraction = ((data_eval.sum()/data_eval.size) - misid)/(efficiency-misid)
    '''
    deff = np.sqrt(efficiency*(1-efficiency)/k_eval.size)
    dmisid = np.sqrt(misid*(1-misid)/pi_eval.size)
    dfrac = np.sqrt((dmisid*((data_eval.sum()/data_eval.size-misid)-efficiency))**2 +
                    + (deff*((data_eval.sum()/data_eval.size-misid)-misid))**2)/(efficiency-misid)**2
    '''

    df_stat = stat_error(fraction, data_eval.size, efficiency, misid)

    df_syst = syst_error(
        fraction, (pi_eval.size, k_eval.size), efficiency, misid)

    if print_tree:
        if print_tree.endswith('.txt') is not True:
            print_tree += '.txt'
        tree_diagram = tree.export_text(dtc, feature_names=vars)
        with open(print_tree, 'w') as file:
            file.write(f'Number of nodes = {n_nodes} \n')
            file.write(f'Max depth = {max_depth} \n \n')
            file.write(tree_diagram)

    print(f'Efficiency is {efficiency} +- {np.sqrt(efficiency*(1-efficiency)/k_eval.size)}\n')
    print(f'Misid is {misid} +- {np.sqrt(misid*(1-misid)/pi_eval.size)}\n')
    print(f'The estimated fraction of K events is {fraction} +- {df_stat} (stat) +- {df_syst} (syst)\n')

    fr = (fraction, df_stat, df_syst)

    algorithm_parameters = (efficiency, misid)

    '''
    if stat_split:
        subdata = np.array_split(pred_array, stat_split)
        fractions = [data.sum()/len(data) for data in subdata]
        plt.figure('Fraction distribution for dtc')
        plt.hist(fractions, bins=10, histtype='step')
        plt.axvline(x=pred_array.sum()/len(pred_array), color='green')
        plt.savefig(default_figpath('dtc_distrib')) if figpath == '' \
            else plt.savefig(figpath+'/dtc_distrib.png')
        stat_err = np.sqrt(np.var(fractions, ddof=1))
        # print(f"Mean = {np.mean(fractions, dtype=np.float64)}")
        # print(f"Sqrt_var = {stat_err}")
        fr = fr + (stat_err,)
    '''

    return fr, algorithm_parameters


"""
    dtr = tree.DecisionTreeRegressor()
    dtr = dtr.fit(X_train, y_train)

    r_pred_array, r_pi_eval, r_k_eval = dtr.predict(data_set), dtr.predict(pi_test), dtr.predict(k_test)

    r_efficiency = (r_k_eval == 1).sum()/r_k_eval.size
    r_misid = (r_pi_eval == 1).sum()/r_pi_eval.size

    print(f'Regressor Efficiency = {r_efficiency}')
    print(f'Regressor Misidentification probability = {r_misid}')
    print(f'The Regressor predicted K fraction is : {r_pred_array.sum()/len(r_pred_array)}')

    plt.hist(r_k_eval,bins=3000,histtype='step', color='blue')
    plt.hist(r_pi_eval,bins=3000,histtype='step',color = 'red')

"""


if __name__ == '__main__':

    predicted_array, eff, misid = dt_classifier(stat_split=20)

    plt.show()
