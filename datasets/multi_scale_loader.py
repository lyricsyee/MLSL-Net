import numpy as np
import os
import pickle
from tqdm import tqdm
from glob import glob
import random

import torch
import torchvision.transforms as v_transforms
from torch.utils.data import DataLoader

from utils.lidc_nod import LIDCNoduleDataset
from utils.transform import Normalize, RandomTrainAug, TestMultiCrop, MultiScaleCrop

def trans_label(label, number):
    label_dict = {
        "cal": [[6], [1, 2, 3, 4, 5]], 
        "spi": [[1, 2], [3, 4, 5]], 
        "lob": [[1, 2], [3, 4, 5]], 
        "tex": [[5], [4, 3], [2, 1]], 
        "sph": [[4, 5], [1, 2, 3]], 
        "mar": [[3, 4, 5], [1, 2]], 
        "sub": [[3, 4, 5], [1, 2]], 
        "mal": [[1, 2, 3], [4, 5]] }
    label_list = label_dict[label]
    binary_label = -1
    for i in range(len(label_list)):
        if number in label_list[i]:
            binary_label = i
    assert binary_label != -1
    if label != 'tex':
        return [binary_label]    
    elif binary_label == 1: # pggo=0, mggo=1
        return [0, 1]
    elif binary_label == 2: # pggo=1, mggo=0
        return [1, 0]
    elif binary_label == 0: # pggp=0, mggo=0
        return [0, 0]
    else:
        raise Exception
    

class MultiScaleLIDCLoader:
    def __init__(self, config):
        self.config = config
        self.labels = ["cal", "spi", "lob", "tex", "sph", "mar", "sub", "mal"]
        norm_method = Normalize(bound=[-1000, 400], cover=[0., 1.])
        if self.config.multi_scale_method == 'single_network':   # train by single scale but test by multiple crop patches (scale)
            # if not enable "multi crop" in training stage, set train_xy_size, train_z_size with target integers, and set ms_aug=False
            self.train_trans = v_transforms.Compose([
                norm_method, 
                RandomTrainAug(self.config.train_xy_size, self.config.train_z_size, self.config.resize, ms_aug=self.config.train_multi_crop, uniform=False)
            ])
            # if not enable "multi crop test" method, just set test_xy_size and test_z_size to desire test size, and set flip=False
            self.val_trans = v_transforms.Compose([
                norm_method, 
                TestMultiCrop(xy_size=self.config.test_xy_size, z_size=self.config.test_z_size, resize=self.config.resize, flip=self.config.test_flip)
            ])
        elif self.config.multi_scale_method == 'ensemble_network':   # train and test by multiple scale nodule simultaneously
            self.train_trans = v_transforms.Compose([
                norm_method, 
                MultiScaleCrop(crop_method='random', xy_size=self.config.train_xy_size, z_size=self.config.train_z_size, resize=self.config.resize)
            ])
            self.val_trans = v_transforms.Compose([ 
                norm_method, 
                MultiScaleCrop(crop_method='center', xy_size=self.config.train_xy_size, z_size=self.config.train_z_size, resize=self.config.resize)    
            ])
        else:
            raise NotImplemented

        data_root = self.config.DATA_ROOT
        all_paths = glob(os.path.join(data_root, '*.pkl'))
        
        self.index2data = {}
        self.index2label = {}
        for path in tqdm(all_paths):
            nodule_id = path.split('/')[-1].split('.pkl')[0]
            with open(path, 'rb') as f:
                data = pickle.load(f)
            new_label = []
            for j, lab in enumerate(self.labels):
                cur_label = trans_label(lab, data[lab])
                new_label += cur_label
            self.index2data[nodule_id] = data['matrix']
            self.index2label[nodule_id] = new_label
        self.no_valid = False

    def get_current_loader(self, fold):
        if self.no_valid:
            train_file, valid_file = 'trainval.txt', 'test.txt'
        else:
            train_file, valid_file = 'train.txt', 'valid.txt'
        train_dataset = LIDCNoduleDataset(
            os.path.join(fold, train_file), self.index2data, self.index2label, transforms=self.train_trans)
        train_loader = DataLoader(
            train_dataset, batch_size = self.config.batch_size, shuffle=True, 
            num_workers=16, pin_memory=True, drop_last=True
        )
        self.train_iters = len(train_dataset) // self.config.batch_size

        val_dataset = LIDCNoduleDataset(
            os.path.join(fold, valid_file), self.index2data, self.index2label, transforms=self.val_trans)
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False, 
            num_workers=16, pin_memory=True, drop_last=False
        )
        self.val_iters = len(val_dataset) // self.config.batch_size + 1

        test_dataset = LIDCNoduleDataset(
            os.path.join(fold, 'test.txt'), self.index2data, self.index2label, transforms=self.val_trans)
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=16, pin_memory=True, drop_last=False
        )
        self.test_iters = len(test_dataset) // self.config.batch_size + 1

        return train_loader, val_loader, test_loader

    def finalize(self):
        pass






