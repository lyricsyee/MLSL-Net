import os
import pickle
import random

import torch
import numpy as np

from glob import glob
from pprint import pprint
from tqdm import tqdm

from torch.utils.data import Dataset


class LIDCNoduleDataset(Dataset):
    def __init__(self,
        nodule_index,
        index2nodule,
        index2label,
        transforms=None
    ):

        super(LIDCNoduleDataset, self).__init__()

        self.index2nodule = index2nodule
        self.index2label = index2label
        self.dataset = []
        with open(nodule_index, 'r') as f:
            for line in f:
                item = line.split()[0]
                if item not in index2nodule:
                    print('Nodule Index Not Found: %s'%(item))
                    continue
                self.dataset.append(item)
        self.transforms = transforms

    def __getitem__(self, index):
        nodule_index = self.dataset[index]
        nodule = self.index2nodule[nodule_index]
        s = self.index2label[nodule_index]
        s = torch.tensor(np.asarray(s), dtype=torch.long)
        if self.transforms is not None:
            nodule = self.transforms(nodule)
        return {'data':nodule, 'label':s}

    def __len__(self):
        return len(self.dataset)



