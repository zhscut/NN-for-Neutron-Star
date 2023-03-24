# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pydot_ng as pydot
from keras.models import Model
from keras.models import Sequential #sequential model
from keras.layers.core import Dense, Activation #fully connected layers
from tensorflow.keras.utils import plot_model #Drawing
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dropout
from keras.models import load_model

def NNs_trained(UsePath, line, X_P, num):

    model = load_model(UsePath)# Load the trained model
    result = model.predict(X_P)
    print(result)
    result = np.reshape(result, (line, 2))
    with open(r'G:\NNs\Predictfile\row\result{}.txt'.format(num), 'w') as f:# save path
        np.savetxt(f, result, fmt='%lf')

def NNs(InputPoint, SavePath, Outpoints):
    EPOCH = 800
    BATCH_SIZE = 500
    VERBOSE = 2#Output progress bar records. In case of 2, output one line for each epoch
    INIT_LR = 0.0003
    OPTIMIZER = Adam(INIT_LR)
    VALIDATION_SPLIT = 0.1#Proportion of training set used for verification set
    RESHAPED = InputPoint
    #
    X_train = np.loadtxt(r'G:\NNs\Input.txt')
    X_test = np.loadtxt(r'G:\NNs\Input.txt')
    Y_train = np.loadtxt(r'G:\NNs\Output.txt')
    Y_test = np.loadtxt(r'G:\NNs\Output.txt')
    X_P = np.loadtxt(r'G:\NNs\I.txt')
    X_P = np.reshape(X_P, (1, InputPoint))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_P = X_P.astype('float32')
    #X_P = np.reshape(X_P, (1,60))

    #Layer design

    model = Sequential()
    model.add(Dense(units=InputPoint, input_shape=(RESHAPED,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(Outpoints, activation='tanh'))#output layer, please make sure the activation fuction correctly
    model.summary()#View summary information of NN

    #training method
    model.compile(loss='mse', optimizer= OPTIMIZER)


    # early_stop = EarlyStopping(monitor= 'val_loss', patience= 10, verbose= 2)
    # Reduce = ReduceLROnPlateau(monitor= 'val_loss', factor= 0.1,
    #                            patience= 5,verbose= 2, mode= 'auto',
    #                            epsilon= 0.00005,cooldown=0, min_lr= 0)

    #training
    history = model.fit(X_train, Y_train, epochs= EPOCH, shuffle= True, batch_size= BATCH_SIZE,
                        validation_split= VALIDATION_SPLIT)## callbacks=[early_stop, Reduce]
    #save model
    model.save(SavePath)
    result = model.predict(X_P)
    print(result)
    #result[:, 0] = result[:, 0]
    #result[:, 1] = result[:, 1]

    result = np.reshape(result, (int(Outpoints/2), 2))
    with open(r'G:\NNs\result.txt', 'a') as f:
        np.savetxt(f, result, fmt='%lf')


    plt.plot(history.history['loss'],color= 'r', label= 'Training Loss')
    plt.plot(history.history['val_loss'], color= 'g', label= 'Validation Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.show()
#Plot-related function
# fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# ax[0].plot(history.history['loss'], color='r', label='Training Loss')
# ax[0].plot(history.history['val_loss'], color='g', label='Validation Loss')
# ax[0].legend(loc='best', shadow=True)
# ax[0].grid(True)

# ax[1].plot(history.history['accuracy'], color='r', label='Training Accuracy')
# ax[1].plot(history.history['val_accuracy'], color='g', label='Validation Accuracy'20)
# ax[1].legend(loc='best', shadow=True)
# ax[1].grid(True)
