import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


class MultiLabelSoftmax(nn.Module):
    def __init__(self, gamma_pos=1., gamma_neg=1.):
        super(MultiLabelSoftmax, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, outputs, targets):
        targets = targets.float()
        outputs = (1 - 2 * targets) * outputs
        y_pred_neg = outputs - targets * 1e15
        y_pred_pos = outputs - (1 - targets) * 1e15
        zeros = torch.zeros_like(outputs[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        neg_loss = (1 / self.gamma_neg) * torch.log(torch.sum(torch.exp(self.gamma_neg * y_pred_neg), dim=-1))
        pos_loss = (1 / self.gamma_pos) * torch.log(torch.sum(torch.exp(self.gamma_pos * y_pred_pos), dim=-1))

        loss = torch.mean(neg_loss + pos_loss)
        return loss




class BinaryCEL(nn.Module):
    def __init__(self):
        super(BinaryCEL, self).__init__()

    def forward(self, outputs, targets):
        return F.binary_cross_entropy_with_logits(
            outputs, targets.float() )



class HingeCalibratedRanking(nn.Module):
    def __init__(self):
        super(HingeCalibratedRanking, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, targets, reduce=True):
        loss_op = []
        for i in range(outputs.size(0)):
            positive = torch.masked_select(outputs[i], targets[i].byte())
            negative = torch.masked_select(outputs[i], (1-targets[i]).byte())

            if negative.size(0) != 0:
                neg_calib = F.relu(1 + negative).mean()
            else:
                neg_calib, negative = 0., 0.
            if positive.size(0) != 0:
                pos_calib = F.relu(1 - positive).mean()
            else:
                pos_calib, positive = 0., 0.
            hinge = 1. + negative.unsqueeze(-1) - positive
            l_hinge = F.relu(hinge).mean()
            loss_op.append(l_hinge + neg_calib + pos_calib)

        loss_op = torch.stack(loss_op, dim=0)
        return loss_op.mean()



