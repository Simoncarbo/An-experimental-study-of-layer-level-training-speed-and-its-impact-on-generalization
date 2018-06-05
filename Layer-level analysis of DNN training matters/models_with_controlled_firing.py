from keras.models import Model
from keras.engine import Layer
import keras.backend as K

import tensorflow as tf


class ReLUWithControlledFiring(Layer):
    '''
    Relu layer where relu's regime (identity or zero) can be either determined by the input's sign (classical way of doing) or by a binary placeholder given as input to the network
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.firing_fixed = K.variable(False, dtype='bool')
        
    def build(self, input_shape):
        super().build(input_shape)
        
        self.firing = K.placeholder(shape=input_shape)
        self.built = True
        
    def call(self, inputs):
        return tf.cond(self.firing_fixed, lambda: inputs * self.firing, lambda: K.relu(inputs))
    
    def compute_output_shape(self, input_shape):
        return input_shape


class ModelWithControlledFiring(Model):
    '''
    Class for models containing ReLUWithControlledFiring layers
    '''
    def __init__(self, inputs, outputs, name=None):
        super().__init__(inputs,outputs, name = name)
        
        self.relu_layers = []
        self.firing_placeholders = []
        for l in self.layers:
            if type(l).__name__ == 'ReLUWithControlledFiring':
                self.relu_layers.append(l)
                
                # collect placeholders that should contain the firing of the neurons
                self.firing_placeholders.append(l.firing)
        
        self._feed_inputs += self.firing_placeholders
        self._feed_input_shapes += [f._keras_shape for f in self.firing_placeholders]
        self._feed_input_names += [f.name for f in self.firing_placeholders]
        
        self.firing_fixed = False
        for l in self.relu_layers:
            K.set_value(l.firing_fixed,False)
        
            
    def set_firing_mode(self,firing_fixed):
        for l in self.relu_layers:
            K.set_value(l.firing_fixed,firing_fixed)
        
        self.firing_fixed = firing_fixed
                
    def get_firing(self,x_train, batch_size = 32): 
        current_firing_mode = self.firing_fixed
        self.set_firing_mode(False)
        
        # creates model that returns the activations
        preacts_model = ModelWithControlledFiring(inputs = self.inputs, outputs = [l.input for l in self.relu_layers])
        
        preacts = preacts_model.predict(x_train, batch_size = batch_size)
        firing = [preact >0 for preact in preacts]
        
        # reset initial firing mode
        self.set_firing_mode(current_firing_mode)
        return firing       