"""

"""
from sklearn import tree
import numpy as np
from data.import_functions import get_txtpaths
from sklearn.model_selection import train_test_split

def dt_classifier(txt_names = ['train_array_prova.txt', 'data_array_prova.txt'], txt_path='../data/txt'):
    """
    """

    # Perché questo lavoro con le directories si fa qua e non nel main? Risulta
    # necessario o comodo per qualcosa di specifico?

    mc_array_path, data_array_path = get_txtpaths(filenames=txt_names, rel_path=txt_path)

    mc_set , data_set = np.loadtxt(mc_array_path), np.loadtxt(data_array_path)

    clf = tree.DecisionTreeClassifier()

    X_train, X_test, y_train, y_test = train_test_split(mc_set[:,:-1], mc_set[:,-1], test_size=0.3, random_state=1)
    clf = clf.fit(X_train, y_train)

    

    pi_test = np.array([X_test[i, :] for i in range(
        np.shape(X_test)[0]) if y_test[i] == 0])
    K_test = np.array([X_test[i, :] for i in range(
        np.shape(X_test)[0]) if y_test[i] == 1])
    
    pred_array, pi_eval, K_eval = clf.predict(data_set), clf.predict(pi_test), clf.predict(K_test)

    efficiency = (K_eval == 1).sum()/K_eval.size
    misid = (pi_eval == 1).sum()/pi_eval.size

    return pred_array, efficiency, misid

if __name__ == '__main__':

    predicted_array, eff, misid = dt_classifier()

    print(f'Efficiency = {eff}')
    print(f'Misidentification probability = {misid}')
    print(f'The predicted K fraction is : {predicted_array.sum()/len(predicted_array)}')

