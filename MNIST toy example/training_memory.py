'''
Keras callbacks for storing input and feedbacks of layers during training
'''

import numpy as np
from numpy import random

from keras.callbacks import Callback
import keras.backend as K
from keras.models import Model

def create_gradient_function(model, tensors, loss, add_batchaxis = False):
    '''
    create gradient function for the specified tensors
    
    tensors is a list of tensors. The gradient will be computed with respect to these tensors.
    
    Taken from discussion in https://github.com/fchollet/keras/issues/2226 
    
    add_batchaxis is used when tensors are parameters. In this case, batches of samples are aggregated when computing gradients.
    To get sample-wise gradients, batchsize = 1 can be used, but a dummy axis needs to be added to parameter gradients.
    '''
    gradients = model.optimizer.get_gradients(loss, tensors)
    
    if add_batchaxis:
        for i,g in enumerate(gradients):
            gradients[i] = K.expand_dims(g,0)
    
    input_tensors = [model.inputs[0], # input data
                     model.sample_weights[0], # how much to weight each sample by
                     model.targets[0], # labels
                     K.learning_phase()] # train or test mode
    

    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    return get_gradients

 
def get_samplewise_smoothgrads(model, tensors, x,y, training_phase = 0., batch_size = 32, verbose = 0, get_gradients = None,
                               sample_size = 50, noise_level = 0.2):
    '''
    extracts sample-wise gradients w.r.t. tensor
    To remove noise, uses technique from SmoothGrad paper
    
    training phase is a binary float: 0. or 1.
    0 is test mode, 1 is training mode
    
    sample_size: number of noisy samples per original sample
    noise_level: std of gaussian noise w.r.t. input range
    '''
    # get_gradients function can be pre-computed
    if not get_gradients:
        get_gradients = create_gradient_function(model, tensors, model.total_loss)
    
    noise_std = noise_level*( x.max() - x.min() )
    
    gradients = [np.zeros(   (x.shape[0],)+tensors[t]._keras_shape[1:]     ) for t in range(len(tensors))]
    for s in range(sample_size):
        if s == 0:
            noise = 0
        else:
            noise = np.random.randn(*x.shape) * noise_std # necessary to unpack shape with * for randn call
        
        # last element of inputs is not sliced in batches thanks to keras :)
        inputs = [x+noise, np.ones(x.shape[0]),y,training_phase]

        gradient_list = model._predict_loop(get_gradients, inputs, batch_size = batch_size, verbose = verbose)
        
        # when there is only one output in _predict_loop function, keras removes the list format and gradient_list is not a list anymore
        if len(tensors) == 1:
            gradients[0] += gradient_list
        else:
            for i,grad in enumerate(gradient_list):
                gradients[i] += grad
                
        #print(s)
            
    for i in range(len(gradients)):
        gradients[i] /= sample_size
    return gradients
    
class LayerFeedbackMemory(Callback):
    '''
    stores gradients of hidden activations along training
    assumes channel_last data format
    '''
    def __init__(self, layer, model, x, y,sample_indices,neuron_indices, epoch_start = 0,epoch_stop = np.inf, batch_size=128, batch_frequency = 30, sample_size = 1, noise_level = 0.):
        '''
        layer: Object of class Layer. Layer at which gradients stats of output activations are computed. 
        '''
        self.sample_indices = sample_indices
        self.neuron_indices = neuron_indices
        
        self.x = x[self.sample_indices]
        self.y = y[self.sample_indices]
        
        self.get_gradients = create_gradient_function(model, [layer.output], model.total_loss)
        self.layer = layer
  
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.noise_level = noise_level
        self.epoch_start = epoch_start
        self.epoch_stop = epoch_stop
        self.batch_frequency = batch_frequency
        
        self.gradients_accumulator = []
        
        self.epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch +=1
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.batch_frequency == 0 and self.epoch_start<=self.epoch<=self.epoch_stop: #batch 0 is accepted

            gradients = get_samplewise_smoothgrads(self.model, [self.layer.output], self.x,self.y, 
                                                   training_phase = 0., get_gradients = self.get_gradients,
                                                   batch_size = self.batch_size,
                                                   sample_size = self.sample_size, noise_level = self.noise_level)[0]

            if len(gradients.shape)==4:
                gradients = gradients.reshape((-1,gradients.shape[-1]))

            self.gradients_accumulator.append(gradients[:,self.neuron_indices].astype(np.float16))

def model_skip(model, output_layers, global_pooling = None, get_output = True):
    '''
    returns new model with skip connections from output_layers to output of model
    output_layers: layer names whose activations will be in model output
    pooled: specifies if output has been gathered through pooling
    get_output: boolean. If true output of the output_layer is extracted. If False, it's the input
    '''
    
    model_layers = [l.name for l in model.layers]
    pooled = [False]*len(output_layers) 
    
    output = []
    for i,l in enumerate(output_layers):
        if l not in model_layers:
            raise ValueError('Layer '+l+' not in standard model')
        
        if get_output:
            tensor = model.get_layer(l).output
        else:
            tensor = model.get_layer(l).input
        if len(tensor._keras_shape) == 4:  # checks if output is a bundle of filtermaps
            if global_pooling == 'max':
                tensor = GlobalMaxPooling2D()(tensor)
            if global_pooling == 'average':
                tensor = GlobalAveragePooling2D()(tensor)
            pooled[i] = True
        output.append(tensor)
        
    return Model(model.input,output), pooled

def model_extract(model, data, output_layers, global_pooling = None, 
                  batch_size = 32, verbose = 0, include_pool_info = False, get_output = True):
    '''
    extract features/representation of data at output_layers
   
    output_layers: layer names whose activations will be in model output
     get_output: boolean. If true output of the output_layer is extracted. If False, it's the input
    '''
    model, pooled = model_skip(model, output_layers, global_pooling, get_output = get_output)
   
    outputs = model.predict(data, batch_size, verbose)
    if not include_pool_info:
        return outputs
    else:
        return outputs,pooled
    
class LayerInputMemory(Callback):
    '''
    assumes channel_last data format
    '''
    def __init__(self, layer, model, x,sample_indices,neuron_indices, 
                 epoch_start = 0,epoch_stop = np.inf,batch_size=128, batch_frequency = 30):
        '''
        layer: Object of class Layer. Layer at which output activations will be stored. 
        '''
        self.sample_indices = sample_indices
        self.neuron_indices = neuron_indices
        
        self.x = x[self.sample_indices]
        
        self.layer = layer
        self.batch_size = batch_size
        self.epoch_start = epoch_start
        self.epoch_stop = epoch_stop
        self.batch_frequency = batch_frequency
        
        self.activations_accumulator = []
        
        self.epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch +=1
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.batch_frequency == 0 and self.epoch_start<=self.epoch<=self.epoch_stop: #batch 0 is accepted

            activations = model_extract(self.model, self.x, [self.layer.name], batch_size = self.batch_size, get_output = False)

            if len(activations.shape)==4:
                activations = activations.reshape((-1,activations.shape[-1]))

            self.activations_accumulator.append(activations[:,self.neuron_indices].astype(np.float16))