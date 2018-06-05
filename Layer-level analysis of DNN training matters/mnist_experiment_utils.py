import os
import numpy as np

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model

from models_with_controlled_firing import ReLUWithControlledFiring, ModelWithControlledFiring

def get_split(nb,x,y):
    '''
    divides data in two sets, one with nb samples per class, the other contains the rest
    '''
    train_selection = set()
    test_selection = set()
    for c in range(y.shape[1]): # iterates on classes
        indexes_class = np.where(y[:,c]==1)[0]
        
        train_selection = train_selection.union(set(np.random.choice(indexes_class,size = nb,replace = False)))
        test_selection  = test_selection.union(set(indexes_class).difference(train_selection))
        
    train_selection = list(train_selection)
    test_selection  = list(test_selection)
    
    x_train,x_test = x[train_selection], x[test_selection]
    y_train,y_test = y[train_selection], y[test_selection]
    
    return x_train, y_train, x_test, y_test

def load_data(experiment):
    '''
    Experiment is an int specifying the index of the experiment (experiments are run multiple times)
    '''
    if not os.path.isfile('data/x_train'+str(experiment)+'.npy'):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        m, st = x_train.mean(), x_train.std()
        x_train -=m
        x_test -=m
        x_train /=st
        x_test /=st

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)


        nb_train = 10 # number of samples per class kept
        x_train, y_train, _,_ = get_split(nb_train,x_train,y_train)

        np.save('data/x_train'+str(experiment)+'.npy',x_train)
        np.save('data/y_train'+str(experiment)+'.npy',y_train)
        np.save('data/x_test'+str(experiment)+'.npy',x_test)
        np.save('data/y_test'+str(experiment)+'.npy',y_test)

        x_train = x_train.reshape(x_train.shape[0], 784)
        x_test = x_test.reshape(x_test.shape[0], 784)
    else:
        x_train=np.load('data/x_train'+str(experiment)+'.npy')
        y_train=np.load('data/y_train'+str(experiment)+'.npy')
        x_test=np.load('data/x_test'+str(experiment)+'.npy')
        y_test=np.load('data/y_test'+str(experiment)+'.npy')

        x_train = x_train.reshape(x_train.shape[0], 784)
        x_test = x_test.reshape(x_test.shape[0], 784)
        
    return x_train, y_train, x_test, y_test



def get_model(experiment):
    '''
    Experiment is an int specifying the index of the experiment (experiments are run multiple times)
    '''
    inp = Input((784,))
    x = Dense(784, activation = 'relu', name = 'dense0')(inp)
    for i in range(1,10):
        x = Dense(784, activation = 'relu', name = 'dense'+str(i))(x)
    x = Dense(10, activation = 'softmax', name = 'densef')(x)
    model = Model(inp,x)
    
    weights_location = 'initial_weights/model_initial_weights'+str(experiment)+'.h5'
    if not os.path.isfile(weights_location):
        model.save_weights(weights_location)
    else:
        model.load_weights(weights_location)
    return model

def get_model_noFeedbackImprovement(experiment):
    '''
    Experiment is an int specifying the index of the experiment (experiments are run multiple times)
    '''
    inp = Input((784,))
    x = Dense(784, name = 'dense0')(inp)
    x = ReLUWithControlledFiring()(x)
    for i in range(1,10):
        x = Dense(784, name = 'dense'+str(i))(x)
        x = ReLUWithControlledFiring()(x)
    x = Dense(10, activation = 'softmax', name = 'densef')(x)

    model = ModelWithControlledFiring(inp,x)
    
    weights_location = 'initial_weights/model_initial_weights'+str(experiment)+'.h5'
    if not os.path.isfile(weights_location):
        model.save_weights(weights_location)
    else:
        model.load_weights(weights_location)
    return model