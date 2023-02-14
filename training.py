"""
Trains a DNN with a numpy array with variable data columns to distinguish between pions and Kaons given multiple variables (features) on which to train simultaneously
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Model
import time

def deepneuralnetwork(training_array, data_array, neurhid1 = 20, neurhid2 = 10, neurhid3 = 5 , epochnum = 100, plotflag = False, verb = 0, testdata = True):
    """
    """
    np.random.seed(int(time.time()))

    training_set = np.loadtxt(training_array)
 
    #print(training_set)

    pid = training_set[:,-1]
    features = training_set[:,:-1] 

    #print(pid)
    #print(np.max(pid))
    #print(features)

    inputlayer=Input(shape=(np.shape(features)[1],))
    hiddenlayer = Dense(neurhid1, activation='relu')(inputlayer)
    hiddenlayer = Dense(neurhid2, activation='relu')(hiddenlayer)
    hiddenlayer = Dense(neurhid3, activation='relu')(hiddenlayer)
    outputlayer = Dense(1, activation='sigmoid')(hiddenlayer)
    deepnn = Model(inputs=inputlayer, outputs=outputlayer)
    deepnn.compile(loss='binary_crossentropy', optimizer='adam')
    deepnn.summary()

    history = deepnn.fit(features, pid, validation_split=0.5, epochs=epochnum, verbose=verb, batch_size=256)

    plt.figure('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Binary CrossEntropy Loss')

    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.legend()

    data_set = np.loadtxt(data_array)[:,:-1] if testdata else np.loadtxt(data_array)

    pred_array = deepnn.predict(data_set)
    print(data_set)
    f_pred = np.sum(pred_array)
    print(f'The predicted K fraction is : {f_pred/len(pred_array)}')
    print('Max prediction :',np.max(pred_array))
    print('Min prediction :',np.min(pred_array))
    if plotflag : plt.show()

    return pred_array

if __name__ == '__main__':
    predicted = deepneuralnetwork('./data/txt/train_array_prova.txt','./data/txt/data_array_prova.txt', plotflag = True, verb = 0)
