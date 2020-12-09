from torch import nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        
    def _reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

    def _linear_combination(self, x, y, epsilon): 
        return epsilon*x + (1 - epsilon)*y
    
    def forward(self, preds, target):
        """
        Expects an input where dim 1 contains the n_classes
        """
        n_classes = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        loss = self._reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        return self._linear_combination(loss/n_classes, nll, self.epsilon)   