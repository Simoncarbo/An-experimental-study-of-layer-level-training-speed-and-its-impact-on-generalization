'''
Get training parameters such as learning rate schedules and stopping criteria for training of the five tasks
'''
import numpy as np

from experiment_utils import lr_schedule
from layca_optimizers import SGD,Adam,RMSprop,Adagrad

from keras.callbacks import Callback, LearningRateScheduler

class StoppingCriteria(Callback):
    '''
    Callback that stops training before the announced number of epochs when some criteria are met.
    '''
    def __init__(self, not_working=(0.,-1), finished = 0., converged = np.inf):
        '''
        not_working is a tuple (acc,nbepochs) with the accuracy that should be reached after nbepochs to consider the training as working
        finished is a training loss value for which the training can be considered as finished
        converged is the number of epochs with unchanged training loss which indicates that the network doesn't change anymore
        '''
        super().__init__()
        self.acc, self.nbepochs = not_working
        self.finished = finished
        self.converged = converged
        
        self.previous_loss = -1
        self.counter = 0
        
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch ==self.nbepochs and logs.get('acc')<= self.acc:
            self.model.stop_training = True
        
        if logs.get('loss')<=self.finished:
            self.model.stop_training = True
        
        if logs.get('loss') == self.previous_loss:
            self.counter += 1
            if self.counter >= self.converged:
                self.model.stop_training = True
        else:
            self.counter = 0
            self.previous_loss = logs.get('loss')
            

def get_training_schedule(task,lr,add = 0):     
    '''
    get number of epochs and learning rate schedule for the three baseline tasks
    '''
    if task == 'C10-CNN1':
        return 100+add, LearningRateScheduler(lr_schedule(lr,0.2,[80+add,90+add,97+add]))
    elif task == 'C100-resnet':
        return 100+add, LearningRateScheduler(lr_schedule(lr,0.1,[70+add,90+add,97+add]))
    elif task == 'tiny-CNN':
        return 80+add, LearningRateScheduler(lr_schedule(lr,0.2,[70+add]))  

def get_training_schedule_sot(task,optimizer,add = 0):
    '''
    get number of epochs and learning rate schedule for the two s.o.t. tasks
    '''
    if optimizer in ['SGD','SGD_weight_decay']:
        if task == 'C10-CNN2':
            return 250, LearningRateScheduler(lr_schedule(0.5,0.5,[i*25 for i in range(1,100)]))
        elif task == 'C100-WRN':
            return 200, LearningRateScheduler(lr_schedule(0.1,0.2,[60,120,160]))
    elif optimizer == 'RMSprop_weight_decay' and task == 'C10-CNN2':
        return 250, LearningRateScheduler(lr_schedule(0.0003,0.5,[i*25 for i in range(1,100)]))
    elif optimizer == 'Adam_weight_decay' and task == 'C100-WRN':
        return 200, LearningRateScheduler(lr_schedule(0.0003,0.2,[60,120,160]))
    elif 'layca' in optimizer:
        lr = 3**-5 if optimizer in ['Adam_layca','SGD_AMom_layca'] else 3**-3
        return 250, LearningRateScheduler(lr_schedule(lr,0.2,[100,170,220]))
    
    
def get_stopping_criteria(task):
    if task == 'C10-CNN1':
        return StoppingCriteria(not_working=(0.2,7), finished = 1e-4, converged = 3)
    elif task == 'C100-resnet':
        return StoppingCriteria(not_working=(0.1,7), finished = 1e-3, converged = 3)
    elif task == 'tiny-CNN':
        return StoppingCriteria(not_working=(0.2,10), finished = 1e-3, converged = 3)
    
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
    
def get_learning_rate_multipliers(model,alpha = 0):
    '''
    provides a dictionary (layer name, lr multiplier) as parametrized by alpha (cfr. Section 4.1)
    '''
    # get layer names in forward pass ordering (layers that are close to input go first)
    layer_names = get_kernel_layer_names(model)
    
    if alpha>0.:
        mult = (1-alpha)**(5/(len(layer_names)-1))
        multipliers = dict(zip(layer_names,[mult**(len(layer_names)-1-i) for i in range(len(layer_names))]))
    elif alpha<=0.:
        mult = (alpha+1)**(5/(len(layer_names)-1))
        multipliers = dict(zip(layer_names,[mult**i for i in range(len(layer_names))]))
    
    return multipliers

def get_optimizer(optimizer, lr, multipliers = {'sqfqzé':1.}):
    '''
    helper function to get a certain optimizer from a string describing it
    multipliers have only been implemented for SGD
    '''    
    if optimizer[:8] == 'SGD_AMom':
        return SGD(lr, layca = 'layca' in optimizer, momentum = 0.9, adam_like_momentum = True, multipliers = multipliers)
    elif optimizer[:3] == 'SGD':
        return SGD(lr, layca = 'layca' in optimizer, multipliers = multipliers)
    elif optimizer[:7] == 'RMSprop':
        return RMSprop(lr, layca = 'layca' in optimizer)
    elif optimizer[:4] == 'Adam':
        return Adam(lr, layca = 'layca' in optimizer)
    elif optimizer[:7] == 'Adagrad':
        return Adagrad(lr, layca = 'layca' in optimizer)