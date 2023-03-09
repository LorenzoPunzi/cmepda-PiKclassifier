"""

"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from utilities.utils import default_txtpaths, default_vars, default_rootpaths, default_figpath
from utilities.import_datasets import array_generator
from utilities.exceptions import InvalidSourceError


def dt_classifier(source=('root', default_rootpaths()), root_tree='tree;1',
                  vars=default_vars(), n_mc=560000, n_data=50000,
                  print_tree='printed_dtc', test_size=0.3, min_leaf_samp=1,
                  crit='gini', stat_split = 0):
    """
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
        exit()

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

    pred_array, pi_eval, k_eval = dtc.predict(data_set), \
        dtc.predict(pi_test), dtc.predict(k_test)

    efficiency = (k_eval == 1).sum()/k_eval.size
    misid = (pi_eval == 1).sum()/pi_eval.size

    if print_tree:
        tree_diagram = tree.export_text(dtc, feature_names=vars)
        file = open(print_tree + '.txt', 'w')
        file.write(f'Number of nodes = {n_nodes} \n')
        file.write(f'Max depth = {max_depth} \n \n')
        file.write(tree_diagram)
        file.close()

    print(f'Efficiency = {efficiency}')
    print(f'Misidentification probability = {misid}')
    print(
        f'The predicted K fraction is : {pred_array.sum()/len(pred_array)}')
    
    if stat_split:

        subdata = np.split(pred_array, stat_split)
        fractions = [data.sum()/len(data) for data in subdata]
        plt.figure('Fraction distribution for dtc')
        plt.hist(fractions,bins=20, histtype='step')
        plt.savefig(default_figpath('fractionsdtc'))
    
    return pred_array, efficiency, misid
    



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

    predicted_array, eff, misid = dt_classifier()

    

    plt.show()
