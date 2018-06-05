'''
llc in our code is equivalent to Layca in our paper
'''

from keras.optimizers import Optimizer
import keras.backend as K
from keras.legacy import interfaces

import tensorflow as tf

from keras.optimizers import Optimizer
import keras.backend as K
from keras.legacy import interfaces

import numpy as np

import tensorflow as tf

def get_optimizer(optimizer, training_mode, model, lr):
    '''
    helper function for adaptive gradients experiment
    '''
    if optimizer == 'SGD':
        return SGD_llc(model, lr, training_mode = training_mode)
    elif optimizer == 'SGD_AMom':
        return SGD_llc(model, lr, momentum = 0.9, adam_like_momentum = True, training_mode = training_mode)
    elif optimizer == 'RMSprop':
        return RMSprop_llc(model, lr, training_mode = training_mode)
    elif optimizer == 'Adam':
        return Adam_llc(model, lr, training_mode = training_mode)
    elif optimizer == 'Adagrad':
        return Adagrad_llc(model, lr, training_mode = training_mode)

def llc(p, step, initial_norm, lr):
    '''
    Core idea behind layer-level control of the training procedure.
    Takes the current parameters and the step computed by an optimizer, and 
         - normalizes the step such that the rotation operated on the layer's weights is controlled
         - after the step has been taken, recovers initial norms of the parameters
    '''
    # used to reduce the risk of NaN or zero values when computing tf.norm(step)
    # doesn't change the algorithm since step is normalized anyways
    step = step/K.max(step) 
    
    # projecting step on tangent space of sphere -> orthogonal to p
    step = step - (K.sum(step * p))* p / initial_norm**2
    
    # normalizing step size
    step = step/ (tf.norm(step)) * initial_norm
    step = tf.where(tf.is_nan(step), tf.zeros_like(step), step)
    new_p =  p - lr * step
            
    # recovering norm of the parameter from before the update (= projecting solution on the sphere)
    new_p = new_p / tf.norm(new_p) * initial_norm
    return new_p
            
            
class SGD_llc(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, model, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, multipliers={'$ùµµ':1.}, adam_like_momentum = False, training_mode = 'llc', **kwargs):
        super(SGD_llc, self).__init__(**kwargs)
        #if momentum != 0.:
        #    print('SGD with layerlevel control should not be used with momentum')
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        
        self.adam_like_momentum = adam_like_momentum
        self.multipliers = multipliers
        if training_mode == 'SGD': # compatibility with older version
            training_mode = 'normal'
        self.training_mode = training_mode
        if self.training_mode not in ['llc', 'normal', 'normalized']:
            raise ValueError('Only llc, normal, or normalized are possible training modes when calling SGD_llc')
        
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
            
            # SGD_lr = 1. is used to reduce the risk of NaN or zero values when using Layca
            # doesn't change the algorithm since SGD step is normalized anyways
            SGD_lr = 1. if self.training_mode == 'llc' else new_lr 
            
            if self.adam_like_momentum:
                v = SGD_lr * ( (self.momentum * m) - (1. - self.momentum) * g )
                self.updates.append(K.update(m, v / SGD_lr))
            else:
                v = self.momentum * m - SGD_lr * g  # velocity
                self.updates.append(K.update(m, v))
             
            if self.nesterov:
                step =  self.momentum * v - SGD_lr * g
            else:
                step =  v
            
            if self.training_mode == 'llc':
                new_p = llc(p, -step, self.initial_norms[p], new_lr)
            elif self.training_mode == 'normal':
                new_p =  p + step
            elif self.training_mode == 'normalized':
                step = step/ tf.norm(step)
                step = tf.where(tf.is_nan(step), tf.zeros_like(step), step)
                new_p =  p + new_lr * step

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD_llc, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RMSprop_llc(Optimizer):
    """RMSProp optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).
    This optimizer is usually a good choice for recurrent
    neural networks.
    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, model, lr=0.001, rho=0.9, epsilon=None, decay=0., training_mode = 'llc',
                 **kwargs):
        super(RMSprop_llc, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.rho = K.variable(rho, name='rho')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        
        self.model = model
        self.initial_norms = {}
        self.training_mode = training_mode
    
    def update_initial_norms(self):
        for w in self.model.weights:
            self.initial_norms.update({w:np.linalg.norm(K.get_value(w))})

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        self.update_initial_norms()
        
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            #new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)
            step = lr * g / (K.sqrt(new_a) + self.epsilon)
            
            if self.training_mode == 'llc':
                new_p = llc(p, step, self.initial_norms[p], lr)
            elif self.training_mode == 'normal':
                new_p =  p - step
            elif self.training_mode == 'normalized':
                step = step/ tf.norm(step)
                step = tf.where(tf.is_nan(step), tf.zeros_like(step), step)
                new_p =  p - lr * step
            
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(RMSprop_llc, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class Adam_llc(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, model, lr=0.001, beta_1=0.9, beta_2=0.999, training_mode = 'llc',
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(Adam_llc, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        
        self.model = model
        self.initial_norms = {}
        self.training_mode = training_mode
    
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

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                step = lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                step = lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            
            if self.training_mode == 'llc':
                new_p = llc(p, step, self.initial_norms[p], lr)
            elif self.training_mode == 'normal':
                new_p =  p - step
            elif self.training_mode == 'normalized':
                step = step/ tf.norm(step)
                step = tf.where(tf.is_nan(step), tf.zeros_like(step), step)
                new_p =  p - lr * step

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(Adam_llc, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Adagrad_llc(Optimizer):
    """Adagrad optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """

    def __init__(self, model, lr=0.01, epsilon=None, decay=0., training_mode = 'llc',**kwargs):
        super(Adagrad_llc, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        
        self.model = model
        self.initial_norms = {}
        self.training_mode = training_mode
    
    def update_initial_norms(self):
        for w in self.model.weights:
            self.initial_norms.update({w:np.linalg.norm(K.get_value(w))})

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        self.update_initial_norms()
        
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            new_a = a + K.square(g)  # update accumulator
            self.updates.append(K.update(a, new_a))
            step = lr * g / (K.sqrt(new_a) + self.epsilon)
            
            if self.training_mode == 'llc':
                new_p = llc(p, step, self.initial_norms[p], lr)
            elif self.training_mode == 'normal':
                new_p =  p - step
            elif self.training_mode == 'normalized':
                step = step/ tf.norm(step)
                step = tf.where(tf.is_nan(step), tf.zeros_like(step), step)
                new_p =  p - lr * step
            
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(Adagrad_llc, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))