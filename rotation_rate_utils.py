import numpy as np
from scipy.spatial.distance import cosine

import matplotlib
import matplotlib.pyplot as plt

from keras.callbacks import Callback

def get_kernel_layer_names(model, remove_skips = False):
    '''
    collects name of all dense and conv2D layers of a model.
    remove_skips allows to ignore conv2D layers used as skip connections in ResNets.
    '''
    layer_names = []
    for l in model.layers:
        layer_class = type(l).__name__
        if layer_class in ['Dense'] or  (layer_class in ['Conv2D'] and not remove_skips):
            layer_names.append(l.name)
        elif layer_class in ['Conv2D'] and remove_skips: # remove 1x1 conv layers that perform skip connections in WideResNet
            if l.kernel_size != (1,1):
                layer_names.append(l.name)
    return layer_names

def get_learning_rate_multipliers(model,alpha = 0):
    # get layer names in forward pass ordering (layers that are close to input go first)
    layer_names = get_kernel_layer_names(model)
    
    if alpha>0.:
        mult = (1-alpha)**(5/(len(layer_names)-1))
        multipliers = dict(zip(layer_names,[mult**(len(layer_names)-1-i) for i in range(len(layer_names))]))
    elif alpha<=0.:
        mult = (alpha+1)**(5/(len(layer_names)-1))
        multipliers = dict(zip(layer_names,[mult**i for i in range(len(layer_names))]))
    
    return multipliers

def plot_parameter_distances(distances, ax = None):
    '''
    utility to plot the layer-wise angular distances between current parameters and initial parameters, as measured over training.
    distances is a numpy array with epoch index in first axis, layer index in second axis
    '''
    matplotlib.rcParams.update({'font.size': 20})
    
    # get one color per layer
    cm = plt.get_cmap('viridis')
    cm_inputs = np.linspace(0,1,distances.shape[1])
    
    if not ax:
        ax = plt.subplot(1,1,1)
    for i in range(distances.shape[-1]):
        layer = i
        ax.plot(np.arange(distances.shape[0]+1), [0]+list(distances[:,layer]), label = str(layer), color = cm(cm_inputs[i]))
    y_limits = np.array([1e-3, 1e-2, 1e-1, 1.1])
    y_limit = y_limits[np.where(distances.max() <= y_limits)[0][0]]
    ax.set_ylim([0,y_limit])
    ax.set_xlim([0,distances.shape[0]])
    
    if y_limit == 1.1:
        ax.set_ylim([0,1.])
        ax.set_yticks([0.,1.])
        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine distance')
    #plt.tight_layout()

def layerwise_parameter_distance(model, initial_w):
    '''
    for each layer, computes cosine distance between current weights and initial weights
    initial_w is a list of tuples containing layer name and corresponding initial numpy weights
    '''
    s = []
    for l_name, w in initial_w:
        s.append(cosine( model.get_layer(l_name).get_weights()[0].flatten(), w.flatten()))
    return s

class LayerwiseParameterDistanceMemory(Callback):
    '''
    Computes and saves distance travelled by weights of each layer during training
    '''
    def __init__(self, initial_w, batch_frequency):
        '''
        initial_w is a list of tuples containing layer name and corresponding initial numpy weights
        '''
        super().__init__()
        self.batch_frequency = batch_frequency
        self.initial_w = initial_w
        
        self.memory = []
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.batch_frequency == 0: #batch 0 is accepted, batch starts at 0 at every epoch

            dist = layerwise_parameter_distance(self.model, self.initial_w)

            self.memory.append(dist)