import tensorflow_core as tf 
from tensorflow_core.keras.callbacks import Callback
from tensorflow_core.keras import backend as  K
import math
import numpy as np
import matplotlib.pyplot as plt

class CyclicLR(Callback):

    ''' This is CyclicLR package designed for tensorflow 2.
        Reference is taken from : 
        It only does traingular CyclicLR
    '''

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000, test_data=None, monitor_val=0):
        super(CyclicLR,self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.current_iteration = 0
        self.history = {}
        # epoch_stats will hold the data that is available at the end of an epoch
        self.epoch_stats = {}
        # Batch stats will hold val_acc and val_loss for every batch end, if test_data is given
        self.batch_stats = {}

        self.test_data = test_data
        # If monitor_val is 1, then val_loss and acc will be calculated every iteration
        self.monitor_val = monitor_val


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


        if(self.monitor_val == 1 and self.test_data != None):
            val_loss, val_acc = self.model.evaluate(self.test_data, verbose=0)
            self.batch_stats.setdefault('val_loss', []).append(val_loss)
            self.batch_stats.setdefault('val_acc', []).append(val_acc)

    def on_epoch_end(self,epochs, logs={}):
        ''' Store validation loss and validation accuracy for each epoch '''
        logs = logs or None

        self.epoch_stats.setdefault('epochs', []).append(epochs+1)
        for k, v in logs.items():
            self.epoch_stats.setdefault(k,[]).append(v)

    def plot_lr(self):
        fig,ax = plt.subplots()
        ax.plot(self.history['iterations'],self.history['lr'])
        ax.set_xlabel('Iterations')
        ax.set_ylabel('learning rate')
        plt.show()

    def plot_lr_loss(self, metric='train', loss_threshold=100):
        ''' Plots lr vs loss '''
        lr = []
        f_loss = []
        if(metric == 'train'):
            loss = np.array(self.history['loss'])
            idxs = np.where(loss < loss_threshold)
            lr = np.array(self.history['lr'])[idxs]
            f_loss = loss[idxs]
        elif(metric == 'val'):
            loss = np.array(self.batch_stats['val_loss'])
            idxs = np.where(loss < loss_threshold)
            lr = np.array(self.history['lr'])[idxs]
            f_loss = loss[idxs]

        fig,ax = plt.subplots(figsize=(7,7))
        ax.semilogx(lr, f_loss)
        ax.set_xlabel('learning rate',fontsize=16)
        ax.set_ylabel(f"{metric} loss", fontsize=16)

        plt.show()

    def plot_lr_acc(self, metric='train'):
        ''' Plots lr vs accuracy '''
        acc = []
        lr = self.history['lr']
        if(metric == 'train'):
            acc = self.history['acc']
        elif(metric == 'val'):
            acc = self.batch_stats['val_acc']

        fig,ax = plt.subplots(figsize=(7,7))
        ax.semilogx(lr, acc)
        ax.set_xlabel('learning rate',fontsize=16)
        ax.set_ylabel(f"{metric} acc", fontsize=16)

        plt.show()


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
        loss = np.array(self.epoch_stats['val_loss'])
        idxs = np.where(loss < loss_threshold)
        f_loss = loss[idxs]
        f_acc = np.array(self.epoch_stats['val_acc'])[idxs]
        epochs = np.array(self.epoch_stats['epochs'])[idxs]

        ax1.plot(epochs, f_loss, c='red', label='validation loss')
        ax1.set_xlabel('epochs', fontsize=16)
        ax1.set_ylabel('validation loss', fontsize=16)
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.set_ylabel('validation accuracy', fontsize=16)
        ax2.plot(epochs, f_acc, c='blue', label='validation accuracy')
        ax2.legend()

        plt.show()

    def plot_generalization_error(self,show_loss=True):
        ''' This function plots generalization error i.e, (train_loss - val_loss)'''
        train_loss = np.array(self.epoch_stats['loss'])
        val_loss = np.array(self.epoch_stats['val_loss'])
        generalization_error = val_loss - train_loss
        epochs = np.array(self.epoch_stats['epochs'])

        fig,ax = plt.subplots(figsize=(7,7))
        if(show_loss == True):
            ax.plot(epochs, train_loss, label='train_loss', c='r')
            ax.plot(epochs, val_loss, label='val_loss', c='b')
        ax.plot(epochs, generalization_error, label='generalization_error', c='g')

        ax.set_xlabel('epochs', fontsize=16)
        ax.set_ylabel('error/loss', fontsize=16)

        ax.legend()
        plt.show()




def test():
    ''' Test the working of the cycling learning rate '''
    clr = CyclicLR()
    inputs = tf.keras.layers.Input((10,))
    d1 = tf.keras.layers.Dense(2,activation='softmax')(inputs)
    model = tf.keras.Model(inputs=inputs,outputs=d1)
    model.compile(loss='mse',metrics=['acc'])
    X_train, y_test = np.random.randn(10000,10), np.random.randint(0,2,size=(10000,2))
    hist = model.fit(X_train,y_test,batch_size=1,epochs=2,callbacks=[clr],validation_data=(X_train,y_test))
    print('data' in dir(model))
    #clr.plot_lr_loss()
    print(model.evaluate(X_train, y_test,verbose=0))

if __name__ == '__main__':
    test()
