'''
    This module implements ths Stochastic Gradient Descent with warm restarts.
    referene: https://arxiv.org/pdf/1608.03983.pdf
'''

import tensorflow_core as tf
from tensorflow_core import keras 
from tensorflow_core.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

class SGDR(keras.callbacks.Callback):

    def __init__(self,base_lr=0.001,max_lr=0.006,step_size=1000):
        '''
            min_lr, lowest point of the learning rate
            max_lr, the starting point of the learning rate
            step_size, the number of iterations after which a restart should happen
        '''
        super().__init__()
        self.min_lr = base_lr
        self.max_lr = max_lr
        self.max_iter = step_size
        self.curr_iter = 0 
        self.total_iter = 0
        self.history = {}

    def _reset(self, min_lr=0.001, max_lr=0.006, max_iter=1000):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_iter = max_iter
        self.curr_iteration = 0 
        self.history = {}

    def get_lr(self):
        c = self.curr_iter % self.max_iter
        return self.min_lr + 0.5*(self.max_lr - self.min_lr)*(1+np.cos((c/self.max_iter)*np.pi))

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.curr_iter += 1
        K.set_value(self.model.optimizer.lr, self.get_lr())

        self.history.setdefault(
            'lr',[]
        ).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault(
            'iterations',[]
        ).append(self.curr_iter)

        for k, v in logs.items():
            self.history.setdefault(k,[]).append(v)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if "val_acc" in logs.keys():
            self.history.setdefault('val_acc',[]).append(logs['val_acc'])
        if "val_loss" in logs.keys():
            self.history.setdefault('val_loss',[]).append(logs['val_loss'])

    def plot_lr(self):
        fig,ax = plt.subplots()
        ax.plot(self.history['iterations'],self.history['lr'])
        ax.set_xlabel('Iterations')
        ax.set_ylabel('learning rate')
        plt.show()

    def plot_lr_loss(self, loss_threshold=100):
        ''' Plots lr vs loss '''
        loss = np.array(self.history['loss'])
        idxs = np.where(loss < loss_threshold)
        lr = np.array(self.history['lr'])[idxs]
        f_loss = loss[idxs]
        print(f_loss.max())
        plt.semilogx(lr,f_loss)

    def plot_lr_acc(self):
        ''' Plots lr vs accuracy '''
        plt.semilogx(self.history['lr'], self.history['acc'])


    def plot_train_loss_acc(self, loss_threshold=10):
        ''' Plots lr, training loss and acc all in the same plot '''
        fig,ax1 = plt.subplots(figsize=(6,6)) 
        loss = np.array(self.history['loss'])
        idxs = np.where(loss < loss_threshold)
        f_lr = np.array(self.history['lr'])[idxs]
        f_loss = loss[idxs]
        f_acc = np.array(self.history['acc'])[idxs]

        ax1.semilogx(f_lr,f_loss, c='red', label='training loss')
        ax1.set_xlabel('learning rate', fontsize=16)
        ax1.set_ylabel('training loss', fontsize=16)
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.set_ylabel('training accuracy', fontsize=16)
        ax2.semilogx(f_lr, f_acc, c='blue', label='training accuracy')
        ax2.legend()

        plt.show()
    
    def plot_val_loss_acc(self, loss_threshold=10):
        ''' Plots validation loss and accuracy for every epoch '''
        fig,ax1 = plt.subplots(figsize=(6,6)) 
        loss = np.array(self.history['val_loss'])
        idxs = np.where(loss < loss_threshold)
        f_loss = loss[idxs]
        f_acc = np.array(self.history['val_acc'])[idxs]
        epochs = list(range(1,len(f_loss)+1))

        ax1.plot(epochs, f_loss, c='red', label='validation loss')
        ax1.set_xlabel('epochs', fontsize=16)
        ax1.set_ylabel('validation loss', fontsize=16)
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.set_ylabel('validation accuracy', fontsize=16)
        ax2.plot(epochs, f_acc, c='blue', label='validation accuracy')
        ax2.legend()

        plt.show()

def test():
    ''' Test the working of the SGDR learning rate '''
    clr = SGDR()
    inputs = tf.keras.layers.Input((10,))
    d1 = tf.keras.layers.Dense(2,activation='softmax')(inputs)
    model = tf.keras.Model(inputs=inputs,outputs=d1)
    model.compile(loss='mse',metrics=['acc'])
    X_train, y_test = np.random.randn(50000,10), np.random.randint(0,2,size=(50000,2))
    hist = model.fit(X_train,y_test,batch_size=32,epochs=5,callbacks=[clr],validation_split=0.3)
    clr.plot_lr()
    print(clr.history.keys())

if __name__ == '__main__':
    test()
