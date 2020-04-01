"""
Cyclic LR
----------

-> An Ongoing project of mine to build cyclic LR and Onecycle LR from scratch.
I will keep adding new stuff to this module so keep vubung

"""

from torch.optim.lr_scheduler import _LRScheduler
import math
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision.models import resnet34
import matplotlib.pyplot as plt

class CyclicLR(_LRScheduler):
    def __init__(self,optimizer,
                 base_lr=1e-7,
                 max_lr=0.1,
                 stepsize=2000,
                 last_epoch=-1, # Keep track of number of iterations. Used as batch_index
                 mode='traingular',
                 gamma = 1.0,
                 ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.stepsize = stepsize
        if mode not in ['traingular','exp']:
            raise ValueError("The value of mode can be: traingular or exp")
        else:
            self.mode = mode
            self.gamma = gamma

        if last_epoch == -1:
            for group in self.optimizer.param_groups:
                group['lr'] = self.base_lr
        super(CyclicLR,self).__init__(self.optimizer)

    def get_lr(self):
        """
            - This function calculates the learning rate for each iteration.
            - self.last_epoch is automatically set to 0 by base class _LRScheduler once we 
            initialize the CyclicLR. self.last_epoch will be used as a batch_index or the current_iteration we 
            are in.
        """
        cycle = math.floor(1 + self.last_epoch/(2*self.stepsize))
        x = abs(self.last_epoch/self.stepsize - 2*cycle + 1)
        base_height = (self.max_lr - self.base_lr)* max(0,(1-x))
        if self.mode == 'exp':
            scale_factor = self.gamma ** cycle
        else:
            scale_factor = 1
            pass
        
        lr = self.base_lr + base_height * scale_factor
        lrs = []
        for i in self.optimizer.param_groups:
            lrs.append(lr)
        return lrs

def lr_range_test(
    dataloader,
    model,
    optimizer,
    criterion, 
    base_lr=1e-7,
    max_lr=0.1,
    stepsize=2000,
    max_iter=2000,
    device='cuda',
    validate=False,
    log_plot='true',

    ):

    cyclic_lr = CyclicLR(optimizer,base_lr,max_lr,stepsize)
    model = model.to(device)
    train_losses = []
    train_lrs = []
    train_accuracy = []
    t = tqdm(range(max_iter))
    for i in t:
        optimizer.zero_grad()
        data,labels = next(iter(dataloader))
        data, labels = data.to(device), labels.to(device)
        out = model(data)
        loss = criterion(out,labels.squeeze())
        loss.backward()
        optimizer.step()
        cyclic_lr.step()
        losses.append(loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])
        accuracy.append(accuracy_score(
            labels.detach().cpu(),
            out.detach().cpu().argmax(axis=1).squeeze()
        ))
        t.set_postfix(loss=round(losses[-1],2),
                      accuracy=round(accuracy[-1],2),
                      lr=optimizer.param_groups[0]['lr'],)

        if validate == True:
            data
    
    try:
        fig,ax = plt.subplots(2)
        ax[0].plot(lrs,losses)
        ax[1].plot(lrs,accuracy)
    except:
        print("Error While Plotting")
    return lrs,losses,accuracy

if __name__ == "__main__":
    optimizer = Adam(resnet34().parameters())
    clr = CyclicLR(optimizer,
                base_lr = 1e-7,
                max_lr = 0.1,
                )
    
    lrs = []
    for i in range(10000):
        clr.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    plt.plot(lrs)

