''' SGD with warm restarts '''
''' reference: https://arxiv.org/pdf/1608.03983.pdf '''
import numpy as np
import matplotlib.pyplot as plt


def get_lr(min_lr, max_lr, Tcur, Ti):
    ''' Returns learning_rate after emulating the warm restart of SGD '''
    ''' Tcur, refers to the current iteration the training is in. '''
    ''' Ti, refers to the iteration at which the restart should occur '''
    Tcur = Tcur % Ti
    return min_lr + 0.5*(max_lr - min_lr)*(1+np.cos((Tcur/Ti)*np.pi))


lr = []
for i in range(2000):
    lr.append(get_lr(0.01,0.06,i,500))

plt.plot(lr)
plt.show()