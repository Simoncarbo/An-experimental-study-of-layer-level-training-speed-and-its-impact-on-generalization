from keras.optimizers import Optimizer
import keras.backend as K
from keras.legacy import interfaces

import numpy as np

import tensorflow as tf
            
class LARS(Optimizer):
    """
    LARS optimizer (ref: https://arxiv.org/abs/1708.03888)
    
    coded by modifying Keras' SGD optimizer code
    """

    def __init__(self, model, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, multipliers={'$ùµµ':1.}, **kwargs):
        super().__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        
        self.multipliers = multipliers
        self.model = model
        self.initial_norms = {}
        
    def update_initial_norms(self):
        for w in self.model.weights:
            self.initial_norms.update({w:np.linalg.norm(K.get_value(w))})

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        self.update_initial_norms()
        
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):            
            processed = False
            for key in self.multipliers.keys():
                if key in p.name and not processed:
                    new_lr = lr * self.multipliers[key]
                    processed = True
            if not processed:
                new_lr = lr
            
            v = self.momentum * m - new_lr * g  # velocity
            self.updates.append(K.update(m, v))
             
            if self.nesterov:
                step =  self.momentum * v - new_lr * g
            else:
                step =  v
            
            # Apply LARS
            norm = tf.norm(p)
            step = step/ (tf.norm(step)) * norm
            step = tf.where(tf.is_nan(step), tf.zeros_like(step), step)
            new_p =  p + new_lr * step
            # Trick for preventing the norm of weights from increasing too much
            diff = 0.0001
            new_norm = tf.norm(new_p)
            new_p = tf.cond(new_norm-norm>diff*self.initial_norms[p],lambda: new_p / tf.norm(new_p)* (norm + self.initial_norms[p] *diff), lambda: new_p)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'multipliers': self.multipliers}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))