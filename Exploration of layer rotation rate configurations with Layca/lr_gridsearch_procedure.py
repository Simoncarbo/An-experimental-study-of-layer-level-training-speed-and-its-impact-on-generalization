'''
Procedure used for tuning the initial global learning rate for each alpha value in section 5
'''
import math as m

def get_best_lr(dic):
    '''
    dic is a dictionary mapping learning rates to information about the training using this learning rate (such as training history)
    This function returns the current best learning rate based on the validation accuracy
    '''
    lrs = dic.keys()
    best, best_lr = -1, None
    for lr in lrs:
        perf = max(dic[lr]['history']['history']['val_acc'])
        if perf >best:
            best, best_lr = perf, lr
    return best_lr
            
def get_next_lrs(dic, alpha):
    '''
    Learning rate grid search manager. 
    For current results (dic) and the studied alpha value, find the next learning rates to be tried.
    Should be called until the returned list is empty for all alphas
    
    Grid search procedure is explained in Supplementary Material of the paper, section B.2
    '''
    # check if optimal lr is the smallest/highest that has been tried yet for alpha.
    # if it is, try the next smaller/higher lr
    lrs = sorted(dic[alpha].keys())
    best_lr = get_best_lr(dic[alpha])
    if best_lr == lrs[0]:
        return [best_lr/3.]
    elif best_lr == lrs[-1]:
        return [best_lr*3.]
    else:
        # After first check is done, check if best_lr of previous or next alpha is different. 
        # If yes, increase precision of gridsearch appropriately
        alphas = [-0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
        index = alphas.index(alpha)
        neighbour_alphas = [alphas[i] for i in [max(index-1,0),min(index+1,len(alphas)-1)]]
        neighbour_best_lrs = [get_best_lr(dic[a]) for a in neighbour_alphas if any([m.isclose(l,best_lr) for l in dic[a].keys()])]
        for lr in neighbour_best_lrs:
            if (m.isclose(lr,best_lr/3.) or m.isclose(lr, best_lr/3.**0.5) ) and not any([m.isclose(l,best_lr/3.**0.5) for l in lrs]):
                return [best_lr/3.**0.5]
            elif (m.isclose(lr,best_lr*3.) or m.isclose(lr, best_lr*3.**0.5) ) and not any([m.isclose(l,best_lr*3.**0.5) for l in lrs]):
                return [best_lr*3.**0.5]
        return []