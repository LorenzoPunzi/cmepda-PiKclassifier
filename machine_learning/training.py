"""
Trains a DNN with a numpy array with variable data columns to distinguish between pions and Kaons given multiple variables (features) on which to train simultaneously
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Model


def train_dnn(training_array, neurons = [20,10,5], epochnum=100, plotflag=False, verb=0, template_eval = False):
    """
    """
    np.random.seed(int(time.time()))

    training_set = np.loadtxt(training_array)
    pi_set = np.array([training_set[i,:] for i in range(np.shape(training_set)[0]) if training_set[i,-1]== 0])
    K_set = np.array([training_set[i,:] for i in range(np.shape(training_set)[0]) if training_set[i,-1]== 1])
    pid = training_set[:, -1]
    features = training_set[:, :-1]
    # print(pid)
    # print(np.max(pid))
    # print(features)

    inputlayer = Input(shape=(np.shape(features)[1],))
    hiddenlayer = Dense(neurons[0], activation='relu')(inputlayer)
    for i in neurons[1:]:
        hiddenlayer = Dense(i, activation='relu')(hiddenlayer)
    outputlayer = Dense(1, activation='sigmoid')(hiddenlayer)
    deepnn = Model(inputs=inputlayer, outputs=outputlayer)
    deepnn.compile(loss='binary_crossentropy', optimizer='adam')
    deepnn.summary()

    history = deepnn.fit(features, pid, validation_split=0.5,
                         epochs=epochnum, verbose=verb, batch_size=256)

    plt.figure('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Binary CrossEntropy Loss')

    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.legend()
    if plotflag:
        plt.show()
    plt.savefig(os.path.join('fig', "epochs.pdf"))

    return deepnn



if __name__ == '__main__':
    current_path = os.path.dirname(__file__)
    txt_path = '../data/txt'
    train_array_path = os.path.join(
        current_path, txt_path, 'train_array_prova.txt')
    data_array_path = os.path.join(
        current_path, txt_path, 'data_array_prova.txt')

    deepnn = train_dnn(train_array_path, plotflag=True, verb=0)

    testdata = True
    data_set = np.loadtxt(data_array_path)[
                          :, :-1] if testdata else np.loadtxt(data_array_path)

    pred_array = deepnn.predict(data_set)
    print(data_set)
    f_pred = np.sum(pred_array)
    print(f'The predicted K fraction is : {f_pred/len(pred_array)}')
    print('Max prediction :', np.max(pred_array))
    print('Min prediction :', np.min(pred_array))

    # return pred_array
