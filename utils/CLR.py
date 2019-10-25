import tensorflow as tf 
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as  K
import math
import numpy as np
import matplotlib.pyplot as plt

class CyclicLR(Callback):

    ''' This is CyclicLR package designed for tensorflow 2.
        Reference is taken from : 
        It only does traingular CyclicLR
    '''

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000):
        super(CyclicLR,self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.current_iteration = 0
        self.history = {}

    def local_cycle(self):
        ''' local_cycle refers to the current cycle we are in based on epoch_counter '''
        ''' epoch_counter to be used as the current iteration we are at and not the current epoch'''
        return math.floor(1 + self.current_iteration/(2 * self.step_size))

    def local_x(self,cycle):
        ''' At what stage is the iteration iteration_counter in that cycle '''
        return np.abs(self.current_iteration/self.step_size - 2 * cycle + 1)

    def local_lr(self, x):
        ''' What should be the learning rate at iteration x in that cycle '''
        return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))

    def clr(self):
        ''' Return's the lr for the current iteration '''
        cycle = self.local_cycle()
        x = self.local_x(cycle)
        lr = self.local_lr(x)
        return lr

    def on_train_begin(self, logs):
        ''' Settings to set up at the beginning of the training '''

        if self.current_iteration == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}

        self.current_iteration += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault(
            'lr',[]
        ).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault(
            'iterations',[]
        ).append(self.current_iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def plot_lr(self):
        fig,ax = plt.subplots()
        ax.plot(self.history['iterations'],self.history['lr'])
        ax.set_xlabel('Iterations')
        ax.set_ylabel('learning rate')
        plt.show()
        

def test():
    ''' Test the working of the cycling learning rate '''
    clr = CyclicLR()
    inputs = tf.keras.layers.Input((10,))
    d1 = tf.keras.layers.Dense(2,activation='softmax')(inputs)
    model = tf.keras.Model(inputs=inputs,outputs=d1)
    model.compile(loss='mse')
    X_train, y_test = np.random.randn(50000,10), np.random.randint(0,2,size=(50000,2))
    hist = model.fit(X_train,y_test,batch_size=32,epochs=10,callbacks=[clr])
    clr.plot_lr()
