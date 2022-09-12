import torch
import numpy as np
import random
import math 

import torch.nn.functional as F
import torchvision.transforms as transforms


class Normalize(object):
    def __init__(self, bound=[-1000.0, 400.0], cover=[0.0, 1.0]):
        self.minbound = min(bound)
        self.maxbound = max(bound)
        self.target_min = min(cover)
        self.target_max = max(cover)

    def __call__(self, x):
        out = (x - self.minbound) / (self.maxbound - self.minbound)
        out[out>self.target_max] = self.target_max
        out[out<self.target_min] = self.target_min
        return out

class RandomTrainAug:
    def __init__(self, xy_size, z_size, resize, ms_aug=True, uniform=False):
        self.xy_size = xy_size if isinstance(xy_size, list) else [xy_size]
        self.z_size = z_size if isinstance(z_size, list) else [z_size]
        self.resize = resize
        self.range = len(self.xy_size)
        self.ms_aug = ms_aug
        if self.ms_aug:
            self.random_choice = np.asarray([1./len(self.xy_size)]*len(self.xy_size))
        else:
            self.randdom_choice = None
        self.uniform = uniform

    def crop(self, data, des_z, des_y, des_x):
        z, y, x = data.shape
        margi, margj, margk = random.randint(-4, 4), random.randint(-4, 4), random.randint(-2, 2)
        x_start, y_start, z_start = int((x - des_x) // 2) - margi, int((y - des_y) // 2) - margj, int((z - des_z) // 2) - margk
        x_end, y_end, z_end = x_start + des_x, y_start + des_y, z_start + des_z
        if x_start < 0: x_start, x_end = 0, des_x
        if x_end > x: x_start, x_end = x - des_x, x
        if y_start < 0: y_start, y_end = 0, dex_y
        if y_end > y: y_start, y_end = y - des_y, y

        out_ = data[ z_start : z_end, y_start : y_end, x_start : x_end ]                
        out_ = out_[np.newaxis, ...]
        out_ = torch.from_numpy(out_.copy()).float()
        out_ = F.interpolate(out_.unsqueeze(0), size=self.resize, mode='trilinear', align_corners=True)
        return out_

    def __call__(self, data):
        if random.random() < 0.5:
            data = np.flip(data, axis=2)
        if random.random() < 0.5:
            data = np.flip(data, axis=1)
        if random.random() < 0.5:
            data = np.flip(data, axis=0)
        axial_rot_num = random.randint(0, 3)
        sag_rot_num = random.randint(0, 1)
        cor_rot_num = random.randint(0, 1)
        data = np.rot90(data, k=axial_rot_num, axes=(1, 2))
        data = np.rot90(data, k=sag_rot_num*2, axes=(0, 1))
        data = np.rot90(data, k=sag_rot_num*2, axes=(0, 2))

        if self.uniform:
            des_z, des_y = round(np.random.uniform(low=14, high=18)), round(np.random.uniform(low=30, high=66))
            return self.crop(data, des_z, des_y, des_y).squeeze(0)
        else:
            index = np.random.choice(np.arange(self.range), p=self.random_choice)
            des_z, des_y = round(np.random.normal(self.z_size[index], 1)), round(np.random.normal(self.xy_size[index], 1.5))
            out_ = self.crop(data, des_z, des_y, des_y)
            return out_.squeeze(0)

class TestMultiCrop:
    def __init__(self, xy_size, z_size, resize, flip=False):
        self.xy_size = xy_size if isinstance(xy_size, list) else [xy_size]
        self.z_size = z_size if isinstance(z_size, list) else [z_size]
        self.resize = resize
        self.flip = flip
        self.ms_crop = True if len(self.xy_size) > 1 else False
        if not self.ms_crop:
            assert not self.flip

    def __call__(self, data):
        z, y, x = data.shape
        img_list = []
        for i in range(len(self.xy_size)):
            des_x, des_y, des_z = self.xy_size[i], self.xy_size[i], self.z_size[i]
            x_start, y_start, z_start = int((x - des_x) // 2), int((y - des_y) // 2), int((z - des_z) // 2)
            out_ = data[z_start:z_start+des_z, y_start:y_start+des_y, x_start:x_start+des_x]
            out_ = out_[np.newaxis, ...]
            out_ = torch.from_numpy(out_.copy()).float()
            out_ = F.interpolate(out_.unsqueeze(0), size=self.resize, mode='trilinear', align_corners=True)
            img_list.append(out_)
            if self.flip:
                img_list.append(torch.flip(out_, dims=[-1]))
        if self.ms_crop:
            img_list = torch.cat(img_list, dim=0)
        else:
            assert len(img_list) == 1
            img_list = img_list[0].squeeze(0)
        return img_list



class MultiScaleCrop(object):
    def __init__(self, xy_size=[32, 48, 64], z_size=[16, 24, 32], 
            resize=[32, 64, 64],  crop_method='center'
    ):
        self.crop_method = crop_method
        self.xy_size = xy_size
        self.z_size = z_size
        self.resize = resize

        if crop_method == 'random':
            self.aug = 'random'
        elif crop_method == 'center':
            self.aug = None
        else:
            raise NotImplementedError
        self.flip = RandomFlip()
        self.rotate = RandomRotation()

    def crop(self, data, des_xy, des_z, random_crop=True):
        z, y, x = data.shape
        margi, margj, margk = 0, 0, 0
        if random_crop:
            des_xy, dez = des_xy, des_z
            margi, margj, margk = random.randint(-4, 4), random.randint(-4, 4), random.randint(-2, 2)
        assert x >= des_xy and y >= des_xy and z >= des_z
        x_start, y_start, z_start = int((x - des_xy) // 2) - margi, int((y - des_xy) // 2) - margj, int((z - des_z ) // 2) - margk
        x_end, y_end, z_end = x_start + des_xy, y_start + des_xy, z_start + des_z
        if x_start < 0: x_start, x_end = 0, des_x
        if x_end > x: x_start, x_end = x - des_x, x
        if y_start < 0: y_start, y_end = 0, dex_y
        if y_end > y: y_start, y_end = y - des_y, y

        out_ = data[
            z_start : z_end, y_start : y_end, x_start : x_end
        ]
        return out_

    def __call__(self, data):
        assert len(data.shape) == 3, 'Input data shape error !'
        outs = []
        for idx in range(3):
            random_crop = True if self.aug is not None else False
            out_ = self.crop(data, self.xy_size[idx], self.z_size[idx], random_crop=random_crop)
            if self.aug is not None:
                out_ = self.flip(out_)
                out_ = self.rotate(out_)
            out_ = out_[np.newaxis, ...]
            out_ = torch.from_numpy(out_.copy()).float()
            out_ = F.interpolate(out_.unsqueeze(0), size=self.resize, mode='trilinear', align_corners=True).squeeze(0)
            outs.append(out_)
        outs = torch.cat(outs, dim=0)

        return outs

class RandomFlip(object):
    def __call__(self, data):
        if len(data.shape) == 3:
            base = 0
        elif len(data.shape) == 4:
            base = 1
        if random.random() < 0.5: 
            data = np.flip(data, base)
        if random.random() < 0.5:
            data = np.flip(data, base+1)
        if random.random() < 0.5:
            data = np.flip(data, base+2)

        return data

class RandomRotation(object):
    def __call__(self, data):
        if len(data.shape) == 3:
            base = 0
        elif len(data.shape) == 4:
            base = 1
        axial_rot_num = random.randint(0, 3)
        sag_rot_num = random.randint(0, 1)
        cor_rot_num = random.randint(0, 1)
        data = np.rot90(data, k=axial_rot_num, axes=(base+1, base+2))
        data = np.rot90(data, k=sag_rot_num*2, axes=(base, base+1))
        data = np.rot90(data, k=sag_rot_num*2, axes=(base, base+2))
        return data


