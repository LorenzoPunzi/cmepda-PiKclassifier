"""
Trains a DNN with a numpy array with variable data columns to distinguish between pions and Kaons given multiple variables (features) on which to train simultaneously
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Model
import time

np.random.seed(int(time.time()))

training_set = np.loadtxt('data/txt/train_array_prova.txt')

print(training_set)

kpid = training_set[:,-1]
features = training_set[:,:-1] 

print(kpid)
print(np.max(kpid))
print(features)

inputlayer=Input(shape=(np.shape(features)[1],))
hiddenlayer = Dense(2, activation='relu')(inputlayer)
hiddenlayer = Dense(5, activation='relu')(hiddenlayer)
hiddenlayer = Dense(10, activation='relu')(hiddenlayer)
outputlayer = Dense(1, activation='sigmoid')(hiddenlayer)
deepnn = Model(inputs=inputlayer, outputs=outputlayer)
deepnn.compile(loss='binary_crossentropy', optimizer='adam')
deepnn.summary()

history = deepnn.fit(features,kpid,validation_split=0.5,epochs=100,verbose=1, batch_size=256)

plt.figure('Losses')
plt.xlabel('Epoch')
plt.ylabel('Binary CrossEntropy Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Training Loss')
plt.legend()

test_set = np.loadtxt('data/txt/data_array_prova.txt')[:,:-1]
pred_array = deepnn.predict(test_set)
print(test_set)
f_pred = np.sum(pred_array)
print(f'The predicted K fraction is : {f_pred/len(pred_array)}')
print('Max prediction :',np.max(pred_array))
print('Min prediction :',np.min(pred_array))
plt.show()