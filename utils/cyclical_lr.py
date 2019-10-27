"""
    This module implements traingular cyclical learning rates.
    reference: https://arxiv.org/abs/1506.01186
"""
import math
import numpy as np
import matplotlib.pyplot as plt

def local_cycle(iteration_counter: int, step_size: int):
    ''' local_cycle refers to the current cycle we are in based on epoch_counter '''
    ''' epoch_counter to be used as the current iteration we are at and not the current epoch'''
    return math.floor(1 + iteration_counter/(2 * step_size))

def local_x(iteration_counter: int,step_size: int,cycle: int):
    ''' At what stage is the iteration iteration_counter in that cycle '''
    return np.abs(iteration_counter/step_size - 2 * cycle + 1)

def local_lr(base_lr, max_lr,x):
    ''' What should be the learning rate at that iteration in that cycle '''
    return base_lr + (max_lr - base_lr) * max(0, (1 - x))

def plot_cyclic_lr(base_lr=0.001, max_lr=0.006, step_size=2000, total_iterations=4000):
    ''' Just to plot the lerning rate through cyclic lr to see how it looks. '''
    lr = []
    for i in range(total_iterations):
        _cycle = local_cycle(i,step_size)
        _x = local_x(i, step_size, _cycle)
        _lr = local_lr(base_lr,max_lr,_x)
        lr.append(_lr)

    fig,ax = plt.subplots()
    ax.plot(lr)
    ax.set_xlabel('iterations')
    ax.set_ylabel('leraning rate')
    plt.show()

plot_cyclic_lr()