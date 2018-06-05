import sys
sys.path.insert(0, "../")

import numpy as np

from experiment_utils import lr_schedule

from keras.callbacks import Callback, LearningRateScheduler

class StoppingCriteria(Callback):
    '''
    
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
            

def get_training_schedule(task,lr,alpha):  # lr's limits depend on the optimizer...          
    add = 0
    if task == 'cifar100':
        if alpha in [-0.8, -0.6, 0.8]:
            add = 0#50
    if task == 'tinyImagenet':
        if alpha in [-0.8, -0.6]:
            add = 0#30
    
    if task == 'cifar10':
        #if lr <=3**-7:
        #    return 80, lr_schedule(lr,0.2,[])
        #elif lr <=3**-5:
        #    return 80, lr_schedule(lr,0.2,[65])
        #else:
        # return 80+add, LearningRateScheduler(lr_schedule(lr,0.2,[50+add,65+add,75+add]))
        return 100+add, LearningRateScheduler(lr_schedule(lr,0.2,[80+add,90+add,97+add]))
    elif task == 'cifar100':
        #if lr <=3**-7:
        #    return 120, lr_schedule(lr,0.2,[105, 115])
        #elif lr <=3**-5:
        #    return 120, lr_schedule(lr,0.2,[90, 105, 115])
        #else:
        return 120+add, LearningRateScheduler(lr_schedule(lr,0.2,[70+add,90+add,105+add,115+add]))
    elif task == 'SVHN':
        return 80, LearningRateScheduler(lr_schedule(lr,0.2,[50,65,75]))
    elif task == 'tinyImagenet':
        return 80+add, LearningRateScheduler(lr_schedule(lr,0.2,[70+add]))
        #return 30+add, LearningRateScheduler(lr_schedule(lr,0.2,[20+add]))    
    
def get_stopping_criteria(task):
    if task == 'cifar10':
        return StoppingCriteria(not_working=(0.2,7), finished = 1e-4, converged = 3)
    elif task == 'cifar100':
        return StoppingCriteria(not_working=(0.1,7), finished = 1e-3, converged = 3)
    elif task == 'SVHN':
        return StoppingCriteria(not_working=(0.2,10), finished = 1e-2, converged = 3)
    elif task == 'tinyImagenet':
        return StoppingCriteria(not_working=(0.2,10), finished = 1e-3, converged = 3)