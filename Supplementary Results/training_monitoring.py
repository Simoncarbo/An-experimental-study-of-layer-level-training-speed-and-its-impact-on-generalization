'''
Keras callbacks for storing signals of interest (Adam's 2nd raw moment estimation, layer rotation rate per epoch,..) during training.
'''

import sys
sys.path.insert(0, "../")

import numpy as np
from scipy.spatial.distance import cosine

import matplotlib
import matplotlib.pyplot as plt

from keras.callbacks import Callback
from keras.engine.training import _make_batches, _slice_arrays
import keras.backend as K
from keras.losses import categorical_crossentropy

from rotation_rate_utils import get_kernel_layer_names, plot_layerwise_angle_deviation_curves, compute_layerwise_angle_deviation

class LayerRotationRateCurves(Callback):
    '''
    Computes and saves rotation performed by each layer during one epoch
    '''
    def __init__(self, epoch_frequency=1):
        '''
        epoch_frequency is the frequency at which the rotation is computed
        '''
        super().__init__()
        self.epoch_frequency = epoch_frequency
        
        self.memory = []
    
    def set_model(self,model):
        super().set_model(model)
        self.layer_names = get_kernel_layer_names(model) 
    
    def on_epoch_begin(self,epoch,logs = None):
        if epoch % self.epoch_frequency == 0: #epoch 0 is accepted
            # previous_w is a list of tuples containing layer name and corresponding initial numpy weights
            self.previous_w = list(zip(self.layer_names,[self.model.get_layer(l).get_weights()[0] for l in self.layer_names]))
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_frequency == 0: #epoch 0 is accepted
            dist = compute_layerwise_angle_deviation(self.model, self.previous_w)

            self.memory.append(dist)

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
    

    get_gradient_function = K.function(inputs=input_tensors, outputs=gradients)
    return get_gradient_function

def get_gradients(model, tensors, x,y, training_phase = 0., batch_size = 32, verbose = 0, get_gradient_function = None, mode = 'param'):
    '''
    extracts gradients w.r.t. tensors; 
    gradients are computed on model.total_loss or according to argument get_gradient_function if provided
    
    If tensors are activations, get gradients provides sample-wise gradients.
    If tensors are parameters, get gradients provides batch-wise gradients
    mode argument specifies if tensors are parameters (mode = 'param') or activations (mode = 'act')
    
    training phase is a binary float: 0. or 1.
    0 is test mode, 1 is training mode
    '''
    if not get_gradient_function:
        get_gradient_function = create_gradient_function(model, tensors, model.total_loss, add_batchaxis = mode=='param')
    
    # last element of inputs is not sliced in batches thanks to keras :)
    inputs = [x, np.ones(x.shape[0]),y,training_phase] 
    
    if not mode=='param' or (mode=='param' and batch_size == 1):
        gradients = model._predict_loop(get_gradient_function, inputs, batch_size = batch_size, verbose = verbose)
    else:
        # need for custom predict loop when output on a batch does not have same size as the batch. 
        # (E.g. there is only one gradient wrt a parameter on an entire batch, and not one gradient per sample)
        num_samples = x.shape[0]
        ins = inputs
        
        unconcatenated_outs = []
        batches = _make_batches(num_samples, batch_size)
        index_array = np.arange(num_samples)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            if ins and isinstance(ins[-1], float):
                # Do not slice the training phase flag.
                ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = _slice_arrays(ins, batch_ids)

            batch_outs = get_gradient_function(ins_batch)
            if not isinstance(batch_outs, list):
                batch_outs = [batch_outs]
            if batch_index == 0:
                for batch_out in batch_outs:
                    unconcatenated_outs.append([])
            for i, batch_out in enumerate(batch_outs):
                    unconcatenated_outs[i].append(batch_out)
            if verbose == 1:
                progbar.update(step + 1)
                
        if len(unconcatenated_outs) == 1:
            gradients = np.concatenate(unconcatenated_outs[0], axis=0)
        else:
            gradients = [np.concatenate(unconcatenated_outs[i], axis=0) for i in range(len(unconcatenated_outs))]
                    
    return gradients

class trainingMemories(Callback):
    '''
    Computes and saves the norm of kernels and kernel gradients in each layer across training
    Loss is assumed to be categorical_crossentropy, model is assumed to have one unique output layer
    '''
    def __init__(self,x,y, batch_frequency=np.inf):
        '''
        batch_frequency is the frequency at which the angle deviations are computed (minimum once per epoch)
        '''
        super().__init__()
        self.x = x #inputs
        self.y = y #outputs
        self.batch_frequency = batch_frequency
        
        self.memory = {'grad_norms':[],'weight_norms':[]}
    
    def set_model(self,model):
        super().set_model(model)
        layer_names = get_kernel_layer_names(model) 
        # get all kernels in topological order (input layers first)
        self.parameters = [model.get_layer(l).kernel for l in layer_names]
        
        loss = categorical_crossentropy(model.targets[0],model.outputs[0])
        self.get_gradient_function = create_gradient_function(model, self.parameters, loss, add_batchaxis = True)
     
    def layerwise_gradientnorm(self,batch_size = 128):
        '''
        Return gradient norms (after gradient averaging over samples) of the kernels of each layer
        '''
        # first dimension of the grads is the number of batches (One gradient is computed per batch)
        grads = get_gradients(self.model, self.parameters, self.x,self.y, training_phase = 1., batch_size = batch_size, mode = 'param',
                              get_gradient_function = self.get_gradient_function)

        # takes L2 norm of gradient on each batch, then averages over batches
        grad_norms = [np.mean(np.sqrt(np.sum(np.square(g), axis = tuple(range(1,len(g.shape)))))) for g in grads]

        return grad_norms
        
    def on_batch_begin(self, batch, logs=None):
        if batch % self.batch_frequency == 0: #batch 0 is accepted, batch resets at 0 at every epoch

            grad_norms = self.layerwise_gradientnorm()
            weight_norms = [np.linalg.norm(K.get_value(p)) for p in self.parameters]

            self.memory['grad_norms'].append(grad_norms)
            self.memory['weight_norms'].append(weight_norms)

def plot_Adam_2nd_moment_memory(memory,epoch=-1, ax = None):
    # first axis is epoch, second layer, third is stats (mean,std)
    data = np.array(memory)[epoch]
    
    if not ax:
        ax = plt.subplot(1,1,1)
        
    ax.errorbar(np.arange(data.shape[0]), data[:,1],np.vstack([data[:,1]-data[:,0],data[:,2]-data[:,1]]), 
                fmt='o',capsize = 3, ms = 4)
    ax.set_yscale('log')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('2nd raw moment')
    
class Adam_2nd_moment_memory(Callback):
    '''
    Stores the layer-wise mean and std of Adam's estimate of the second moment during training
    '''
    def __init__(self, batch_frequency):
        '''
        '''
        super().__init__()
        self.batch_frequency = batch_frequency
        
        self.memory = []
        
    def set_model(self,model):
        super().set_model(model)
        
        # grab the variables corresponding to Adam's estimate of second moments
        nb_params = int((len(self.model.optimizer.weights)-1)/3)
        self.v_variables = self.model.optimizer.weights[1+nb_params:1+2*nb_params]
    
    def on_batch_end(self, batch, logs=None):
        if batch % self.batch_frequency == 0: #batch 0 is accepted, batch starts at 0 at every epoch
            
            layer_stats = []
            for v in self.v_variables:
                v = K.get_value(v)
                layer_stats.append(np.percentile(v,[10,50,90]))
            
            self.memory.append(layer_stats)
         
    def plot(self, epoch = -1, ax = None):
        plot_Adam_2nd_moment_memory(self.memory,epoch=epoch, ax = ax)