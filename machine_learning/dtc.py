"""

"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from utilities.utils import default_txtpaths , default_vars
from utilities.import_datasets import array_generator


def dt_classifier(source = ('txt', default_txtpaths()), n_mc = 560000, n_data = 50000, root_tree = 'tree;1', vars = default_vars(), print_tree = 'printed_dtc', test_size = 0.3):
    """
    """
    # Perché questo lavoro con le directories si fa qua e non nel main? Risulta
    # necessario o comodo per qualcosa di specifico?
    if source[0] == 'txt':
        mc_array_path, data_array_path = source[1] if source[1] else default_txtpaths()
        mc_set , data_set = np.loadtxt(mc_array_path), np.loadtxt(data_array_path)
    elif source[1] == 'root':
        mc_set , data_set = array_generator(rootpaths=source[1], tree=root_tree, vars=vars, n_mc=n_mc, n_data=n_data)
    else:
        print('ERROR: invalid source for dt_classifier')
    

    dtc = tree.DecisionTreeClassifier()

    X_train, X_test, y_train, y_test = train_test_split(mc_set[:,:-1], mc_set[:,-1], test_size=test_size, random_state=1)
    dtc = dtc.fit(X_train, y_train)

    

    pi_test = np.array([X_test[i, :] for i in range(
        np.shape(X_test)[0]) if y_test[i] == 0])
    K_test = np.array([X_test[i, :] for i in range(
        np.shape(X_test)[0]) if y_test[i] == 1])
    
    pred_array, pi_eval, K_eval = dtc.predict(data_set), dtc.predict(pi_test), dtc.predict(K_test)

    efficiency = (K_eval == 1).sum()/K_eval.size
    misid = (pi_eval == 1).sum()/pi_eval.size

    if print_tree:
        tree_diagram = tree.export_text(dtc, feature_names = vars)
        file = open(print_tree + '.txt', 'w')
        file.write(tree_diagram)
        file.close()
    

    #tree.plot_tree(dtc)

    return pred_array, efficiency, misid

if __name__ == '__main__':

    predicted_array, eff, misid = dt_classifier()

    print(f'Efficiency = {eff}')
    print(f'Misidentification probability = {misid}')
    print(f'The predicted K fraction is : {predicted_array.sum()/len(predicted_array)}')

    #plt.show()

