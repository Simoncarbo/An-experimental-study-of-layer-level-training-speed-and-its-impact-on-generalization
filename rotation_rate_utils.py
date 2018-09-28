'''
Utilities related to layer-wise angle deviation curves
'''

import numpy as np
from scipy.spatial.distance import cosine

import matplotlib
import matplotlib.pyplot as plt

from keras.callbacks import Callback
from keras.engine.training import _make_batches, _slice_arrays
import keras.backend as K
from keras.losses import categorical_crossentropy

def get_kernel_layer_names(model):
    '''
    collects name of all layers of a model that contain a kernel in topological order (input layers first).
    '''
    layer_names = []
    for l in model.layers:
        if len(l.weights) >0:
            if 'kernel' in l.weights[0].name:
                layer_names.append(l.name)
    return layer_names

def plot_layerwise_angle_deviation_curves(deviations, ax = None):
    '''
    utility to plot the layer-wise angular distances between current parameters and initial parameters, as measured over training.
    deviations is a list of lists with epoch index in first axis, layer index in second axis, 
        containing the angle deviations for each layer as recorded over training
    '''
    distances = np.array(deviations)
    
    #matplotlib.rcParams.update({'font.size': 20})
    
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

def compute_layerwise_angle_deviation(current_model, initial_w):
    '''
    for each layer, computes cosine distance between current weights and initial weights
    initial_w is a list of tuples containing layer name and corresponding initial numpy weights
    '''
    s = []
    for l_name, w in initial_w:
        s.append(cosine( current_model.get_layer(l_name).get_weights()[0].flatten(), w.flatten()))
    return s

class LayerwiseAngleDeviationCurves(Callback):
    '''
    Computes and saves distance travelled by weights of each layer during training
    '''
    def __init__(self, batch_frequency=np.inf):
        '''
        batch_frequency is the frequency at which the angle deviations are computed (minimum once per epoch)
        '''
        super().__init__()
        self.batch_frequency = batch_frequency
        
        self.memory = []
    
    def set_model(self,model):
        super().set_model(model)
        layer_names = get_kernel_layer_names(model) 
        # initial_w is a list of tuples containing layer name and corresponding initial numpy weights
        self.initial_w = list(zip(layer_names,[model.get_layer(l).get_weights()[0] for l in layer_names]))
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.batch_frequency == 0: #batch 0 is accepted, batch resets at 0 at every epoch

            dist = compute_layerwise_angle_deviation(self.model, self.initial_w)

            self.memory.append(dist)
    
    def plot(self,ax = None):
        plot_layerwise_angle_deviation_curves(self.memory,ax)