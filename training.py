"""
Trains a DNN with a numpy array with ONE data column to distinguish between pions and Kaons given one variable on which to train simultaneously
WILL NEED TO BE GENERALIZED TO MULTIPLE COLUMNS/VARIABLES ARRAYS
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Model
import time

np.random.seed(int(time.time()))

training_set = np.loadtxt('0.4_nparray.txt')

x = training_set[:,0]  # VARIABLE NAMES SHOULD BE CHANGED APPROPRIATELY 
y = training_set[:,1]

inputlayer=Input(shape=(1,))
hiddenlayer = Dense(2, activation='relu')(inputlayer)
hiddenlayer = Dense(5, activation='relu')(hiddenlayer)
hiddenlayer = Dense(5, activation='relu')(hiddenlayer)
outputlayer = Dense(1, activation='sigmoid')(hiddenlayer)
deepnn = Model(inputs=inputlayer, outputs=outputlayer)
deepnn.compile(loss='binary_crossentropy', optimizer='adam')
deepnn.summary()

history = deepnn.fit(x,y,validation_split=0.4,epochs=100,verbose=1)

plt.figure('Losses')
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])

test_set = np.loadtxt('0.5_nparray.txt')[:,0]
pred_array = deepnn.predict(test_set)
print(test_set)
f_pred = np.sum(pred_array)
print(f'The predicted K fraction is : {f_pred/len(pred_array)}')
print('Max prediction :',np.max(pred_array))
print('Min prediction :',np.min(pred_array))
print(pred_array)
#plt.show()