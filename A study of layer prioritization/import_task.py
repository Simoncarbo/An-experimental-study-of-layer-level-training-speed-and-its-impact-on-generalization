import os
import sys
sys.path.insert(0, "../")

from experiment_utils import import_cifar
from rotation_rate_utils import get_kernel_layer_names
from models import VGG, resnet_v1

def import_task(experiment, mode = ''):
    if experiment not in ['cifar10', 'cifar100', 'SVHN', 'tinyImagenet']:
        raise ValueError('Wrong experiment name.')
    
    if experiment == 'cifar10':
        return import_cifar10_task(mode)
    elif experiment == 'cifar100':
        return import_cifar100_task()
    elif experiment == 'SVHN':
        return import_SVHN_task()
    elif experiment == 'tinyImagenet':
        return import_tinyImagenet_task()

def import_cifar10_task(mode = ''):
    x_train, y_train, x_test, y_test = import_cifar()
    
    def get_model():
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
                        kernel_initializer = 'he_uniform')
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
                        kernel_initializer = 'he_uniform')

        weights_location = 'model_initial_weights/cifar10_initial_weights'+mode+'.h5'
        if 'cifar10_initial_weights'+mode+'.h5' not in os.listdir('model_initial_weights'):
            model.save_weights(weights_location)
        else:
            model.load_weights(weights_location)
            
        return model
    
    model = get_model()
    layer_names = get_kernel_layer_names(model)    
    initial_kernels = list(zip(layer_names,[model.get_layer(l).get_weights()[0] for l in layer_names]))
    
    return x_train, y_train, x_test, y_test, get_model, initial_kernels


def import_cifar100_task():
    x_train, y_train, x_test, y_test = import_cifar(dataset = 100)
    
    def get_model():
        # resnet32
        model = resnet_v1((32,32,3),depth = 32, num_classes = 100,
                          use_bias = False,
                          batchnorm_training = False)

        weights_location = 'model_initial_weights/cifar100_initial_weights_resnet32.h5'
        if 'cifar100_initial_weights_resnet32.h5' not in os.listdir('model_initial_weights'):
            model.save_weights(weights_location)
        else:
            model.load_weights(weights_location)
            
        return model

    model = get_model()
    layer_names = get_kernel_layer_names(model)
    initial_kernels = list(zip(layer_names,[model.get_layer(l).get_weights()[0] for l in layer_names]))
    
    return x_train, y_train, x_test, y_test, get_model, initial_kernels

def import_tinyImagenet_task():
    try:
        import sys
        sys.path.insert(0, "/export/home/sicarbonnell/Recherche/_datasets")
        from import_tinyImagenet import import_tinyImagenet
    except:
        raise ImportError('Our code does not provide the utilities to laod the tinyImagenet dataset.')
    
    x_train, y_train, x_test, y_test = import_tinyImagenet()
    
    # a 11 layer deep VGG style network with batchnorm
    def get_model():
        k = 32
        model = VGG(input_shape = x_train.shape[1:],
                    nbstages = 5,
                    nblayers = [2]*5,
                    nbfilters = [1*k,2*k,4*k,8*k,16*k],
                    nbclasses = y_train.shape[1],
                    use_bias = False,
                    batchnorm_training = False, #use_batchnorm = False,
                    kernel_initializer = 'he_uniform',
                    batchnorm_momentum = 0.9)  ### because training sometimes stops after very few epochs (~15)
    
        weights_location = 'model_initial_weights/tinyImagenet_initial_weights_batchnorm.h5'
        if 'tinyImagenet_initial_weights_batchnorm.h5' not in os.listdir('model_initial_weights'):
            model.save_weights(weights_location)
        else:
            model.load_weights(weights_location)
            
        return model
    
    model = get_model()
    layer_names = get_kernel_layer_names(model)
    initial_kernels = list(zip(layer_names,[model.get_layer(l).get_weights()[0] for l in layer_names]))
    
    return x_train, y_train, x_test, y_test, get_model, initial_kernels#, weights_location