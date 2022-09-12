import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedBinaryCEL(nn.Module):
    def __init__(self, cross=False):
        super(WeightedBinaryCEL, self).__init__()
        self.cross = cross
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, targets):
        pos_outs = self.sigmoid(outputs)
        pos_labels = targets.float()

        neg_labels = 1 - targets.float()
        neg_outs = 1 - self.sigmoid(outputs)

        sumP = pos_labels.sum(dim=0)
        sumN = neg_labels.sum(dim=0)

        if self.cross:
            weightP = (sumN) / (sumP + sumN)
            weightN = (sumP) / (sumP + sumN)
        else:
            weightP = (sumP + sumN + 0.1) / (sumP + 0.1)
            weightN = (sumP + sumN + 0.1) / (sumN + 0.1)

        pos_loss = torch.mul(pos_labels, torch.log(pos_outs))
        neg_loss = torch.mul(neg_labels, torch.log(neg_outs))

        fpcls = - weightN*neg_loss.mean(dim=0) - weightP*pos_loss.mean(dim=0)
        loss = fpcls.mean()

        if loss.item() is np.nan:
            import pdb; pdb.set_trace()

        return loss

class BinaryCEL(nn.Module):
    def __init__(self):
        super(BinaryCEL, self).__init__()

    def forward(self, outputs, targets):
        return F.binary_cross_entropy_with_logits(
            outputs, targets.float() )





        
    




