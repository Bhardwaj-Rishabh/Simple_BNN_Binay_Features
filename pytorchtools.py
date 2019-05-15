import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, in_dim, dir_chk='chkpts', dataset='purchase', patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.count = 0
        self.in_dim = in_dim
        self.dir_chk = dir_chk
        self.dataset = dataset

    def __call__(self, val_loss, model, epoch):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter:', self.counter, 'out of', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.count = 0

        self.count += 1
        self.save_checkpoint(val_loss, model, epoch)

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.dir_chk+'/'+str(epoch)+self.dataset+'-'+str(self.in_dim)+'-'+model.name+'.pt')
        self.val_loss_min = val_loss