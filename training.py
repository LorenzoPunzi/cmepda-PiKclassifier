"""
Trains a DNN with a numpy array with ONE data column to distinguish between pions and Kaons given one variable on which to train simultaneously
WILL NEED TO BE GENERALIZED TO MULTIPLE COLUMNS/VARIABLES ARRAYS
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Model

training_set = np.loadtxt('0.5_nparray.txt')

x = training_set[:,0]  # VARIABLE NAMES SHOULD BE CHANGED APPROPRIATELY 
y = training_set[:,1]

inputlayer=Input(shape=(1,))
hiddenlayer1 = Dense(5, activation='relu')(inputlayer)
hiddenlayer2 = Dense(10, activation='relu')(hiddenlayer1)
hiddenlayer3 = Dense(10, activation='relu')(hiddenlayer2)
hiddenlayer4 = Dense(10, activation='relu')(hiddenlayer3)
outputlayer = Dense(1, activation='sigmoid')(hiddenlayer4)
deepnn = Model(inputs=inputlayer, outputs=outputlayer)
deepnn.compile(loss='binary_crossentropy', optimizer='adam')
deepnn.summary()

history = deepnn.fit(x,y,validation_split=0.5,epochs=500,verbose=1)

plt.figure('Losses')
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])

test_set = np.loadtxt('0.4_nparray.txt')[:,0]
pred_array = deepnn.predict(test_set)
print(type(pred_array))
f_pred = np.sum(pred_array)
print(f'The predicted K fraction is : {f_pred/len(pred_array)}')