"""
Trains a DNN with a numpy array with variable data columns to distinguish between pions and Kaons given multiple variables (features) on which to train simultaneously
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Model


def train_dnn(training_set, neurons = [20,10,5], epochnum=100, plotflag=False, verb=0):
    """
    """
    np.random.seed(int(time.time()))

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
                         epochs=epochnum, verbose=verb, batch_size=128)

    if plotflag:
        plt.figure('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Binary CrossEntropy Loss')

        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.legend()
        plt.savefig(os.path.join('fig', "epochs.pdf"))
        plt.show()

    return deepnn


def eval_dnn(dnn, eval_set, plot= [], test_data = False, f_print = False):
    """
    """
    prediction_array = dnn.predict(eval_set) if not test_data else dnn.predict(eval_set[:,:-1])
    
    if plot:
        plotname = plot[0]
        plt.figure(plotname)
        plt.hist(prediction_array, bins=300, histtype='step', color=plot[1])
        plt.savefig('./fig/predict_'+plotname+'.pdf')
        plt.draw()
    
    if f_print:
        f_pred = np.sum(prediction_array)
        print(f'The predicted K fraction is : {f_pred/len(prediction_array)}')
        print('Max prediction :', np.max(prediction_array))
        print('Min prediction :', np.min(prediction_array))

    return prediction_array

def dnn(trainarrayname, dataarrayname, txt_path = '../data/txt', flagged_data = False, plot_pi = ['Templ_eval', 'red'], plot_K = ['Templ_eval', 'blue'], plot_data = ['Dataeval','blue']):
    
    current_path = os.path.dirname(__file__)
    train_array_path = os.path.join(
        current_path, txt_path, trainarrayname)
    data_array_path = os.path.join(
        current_path, txt_path, dataarrayname)

    training_set = np.loadtxt(train_array_path)
    pi_set = np.array([training_set[i,:] for i in range(np.shape(training_set)[0]) if training_set[i,-1]== 0])
    K_set = np.array([training_set[i,:] for i in range(np.shape(training_set)[0]) if training_set[i,-1]== 1])
    data_set = np.loadtxt(data_array_path)

    deepnn = train_dnn(training_set,neurons=[20,20,20], verb=0) 
    pred_array = eval_dnn(deepnn,data_set,plot=plot_data,test_data=flagged_data,f_print=True)
    pi_eval = eval_dnn(deepnn, pi_set,plot=plot_pi,test_data=True)
    K_eval = eval_dnn(deepnn, K_set,plot=plot_K,test_data=True)

    return pi_eval, K_eval, pred_array

if __name__ == '__main__':
    
    pi_eval, K_eval, pred_array = dnn('train_array_prova.txt','data_array_prova.txt',flagged_data=True)   
    plt.show()
    #print(pred_array)

