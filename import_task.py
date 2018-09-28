'''
Imports data and models for the five tasks of the paper (cfr. Table 1).
'''
import os

from experiment_utils import import_cifar
from rotation_rate_utils import get_kernel_layer_names
from models import VGG, resnet_v1, VGG_pytorchBlogStyle, WideResNet

def import_task(experiment, mode = ''):
    if experiment not in ['C10-CNN1', 'C100-resnet', 'tiny-CNN', 'C10-CNN2','C100-WRN']:
        raise ValueError('Wrong experiment name.')
    
    if experiment == 'C10-CNN1':
        return import_cifar10_task(mode)
    elif experiment == 'C100-resnet':
        return import_cifar100_task()
    elif experiment == 'tiny-CNN':
        return import_tinyImagenet_task()
    elif experiment == 'C10-CNN2':
        return import_pytorchVGG_C10()
    elif experiment == 'C100-WRN':
        return import_WRN_task()

def import_cifar10_task(mode = ''):
    x_train, y_train, x_test, y_test = import_cifar()
    
    def get_model(weight_decay = 0.):
        if mode =='':
            # a 25 layers deep VGG-style network with batchnorm
            k = 32
            model = VGG(input_shape = x_train.shape[1:],
                        nbstages = 4,
                        nblayers = [6]*4,
                        nbfilters = [1*k,2*k,4*k,8*k],
                        nbclasses = y_train.shape[1],
                        use_bias = False,
                        batchnorm_training = False,
                        kernel_initializer = 'he_uniform',
                        weight_decay = weight_decay)
        elif mode == 'fast':
            k = 16
            # a 13 layers deep VGG-style network with batchnorm
            model = VGG(input_shape = x_train.shape[1:],
                        nbstages = 4,
                        nblayers = [3]*4,
                        nbfilters = [1*k,2*k,4*k,8*k],
                        nbclasses = y_train.shape[1],
                        use_bias = False,
                        batchnorm_training = False,
                        kernel_initializer = 'he_uniform',
                        weight_decay = weight_decay)

        weights_location = 'saved_weights/initial_weights_cifar10'+mode+'.h5'
        if 'initial_weights_cifar10'+mode+'.h5' not in os.listdir('saved_weights'):
            model.save_weights(weights_location)
        else:
            model.load_weights(weights_location)
            
        return model
    
    return x_train, y_train, x_test, y_test, get_model


def import_cifar100_task():
    x_train, y_train, x_test, y_test = import_cifar(dataset = 100)
    
    def get_model(weight_decay=0.):
        # resnet32
        model = resnet_v1((32,32,3),depth = 32, num_classes = 100,
                          use_bias = False,
                          batchnorm_training = False, weight_decay = weight_decay)

        weights_location = 'saved_weights/initial_weights_cifar100.h5'
        if 'initial_weights_cifar100.h5' not in os.listdir('saved_weights'):
            model.save_weights(weights_location)
        else:
            model.load_weights(weights_location)
            
        return model
    
    return x_train, y_train, x_test, y_test, get_model

def import_tinyImagenet_task():
    try:
        import sys
        sys.path.insert(0, "/export/home/sicarbonnell/Recherche/_datasets")
        from import_tinyImagenet import import_tinyImagenet
    except:
        raise ImportError('Our code does not provide the utilities to laod the tinyImagenet dataset.')
    
    x_train, y_train, x_test, y_test = import_tinyImagenet()
    
    # a 11 layer deep VGG style network with batchnorm
    def get_model(weight_decay=0.):
        k = 32
        model = VGG(input_shape = x_train.shape[1:],
                    nbstages = 5,
                    nblayers = [2]*5,
                    nbfilters = [1*k,2*k,4*k,8*k,16*k],
                    nbclasses = y_train.shape[1],
                    use_bias = False,
                    batchnorm_training = False, #use_batchnorm = False,
                    kernel_initializer = 'he_uniform',
                    batchnorm_momentum = 0.9, ### because training sometimes stops after very few epochs (~15)
                    weight_decay = weight_decay)  
    
        weights_location = 'saved_weights/initial_weights_tinyImagenet.h5'
        if 'initial_weights_tinyImagenet.h5' not in os.listdir('saved_weights'):
            model.save_weights(weights_location)
        else:
            model.load_weights(weights_location)
            
        return model
    
    return x_train, y_train, x_test, y_test, get_model

def import_pytorchVGG_C10():
    x_train, y_train, x_test, y_test = import_cifar()
    
    def get_model(weight_decay = 0.0005):
        model = VGG_pytorchBlogStyle((32,32), 5, [2,2,3,3,3], [64,128,256,512,512],10,weight_decay=weight_decay,
                                     batchnorm_training = False, use_bias = False, kernel_initializer='he_normal')
        

        weights_location = 'saved_weights/initial_weights_C10-CNN2.h5'
        if 'initial_weights_C10-CNN2.h5' not in os.listdir('saved_weights'):
            model.save_weights(weights_location)
        else:
            model.load_weights(weights_location)
            
        return model
    
    return x_train, y_train, x_test, y_test, get_model

def import_WRN_task():
    x_train, y_train, x_test, y_test = import_cifar(dataset = 100)
    
    def get_model(weight_decay=0.0005):
        # parameters for WideResnet model
        k = 10 # widening factor
        N = 4 # number of blocks per stage. Depth = 6*N+4
        dropout = 0.3
        
        # WRN 28 - 10 with dropout 0.3
        model = WideResNet([16*k,32*k,64*k],[N]*3,dropout,weight_decay,nb_classes = 100)

        weights_location = 'saved_weights/initial_weights_C100_WRN.h5'
        if 'initial_weights_C100_WRN.h5' not in os.listdir('saved_weights'):
            model.save_weights(weights_location)
        else:
            model.load_weights(weights_location)
            
        return model
    
    return x_train, y_train, x_test, y_test, get_model