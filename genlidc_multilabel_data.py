import os
from glob import glob
import pickle
import torch
import torch.nn.functional as F
import argparse

import numpy as np
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm




def get_nodule_matrix(images, center):
    shape = images.shape
    axial_scale = 64
    z_scale = 32
    z, y, x= np.round(center).astype(int)
    xmin, ymin, zmin = x - axial_scale, y - axial_scale, z - z_scale
    z_pad, y_pad, x_pad = [0,0], [0,0], [0,0]
    if xmin < 0:
        x_pad[0], xmin = abs(xmin), 0
    if x + axial_scale > shape[2]:
        x_pad[1] = abs(x + axial_scale - shape[2])
    if ymin < 0:
        y_pad[0], ymin = abs(ymin), 0
    if y + axial_scale > shape[1]:
        y_pad[1] = abs(y + axial_scale - shape[1])
    if zmin < 0:
        z_pad[0], zmin = abs(zmin),0
    if z + z_scale > shape[0]:
        z_pad[1] = abs(z + z_scale - shape[0])
    if z_pad != [0,0] or y_pad != [0,0] or x_pad != [0,0]:
        padding = [z_pad, y_pad, x_pad]
        images = np.pad(images, pad_width=padding, mode='constant', constant_values=images.min())
        print('after padding image size:', images.shape, padding)
    nodule_roi = images[zmin:zmin+2*z_scale, ymin:ymin+2*axial_scale, xmin:xmin+2*axial_scale]
    return nodule_roi

def rescale(image, spacing, new_spacing=[1., 1., 1.], mode='nearest'):
    spacing = np.asarray(spacing)
    new_spacing = np.asarray(new_spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape).astype(int)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    # image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    image = torch.from_numpy(image[np.newaxis, np.newaxis, ...]).float()
    if mode != 'nearest':
        kwargs = {'align_corners': True}
    else:
        kwargs = {}
    image = F.interpolate(image, size=new_shape.tolist(), mode=mode, **kwargs)
    image = image.squeeze(0).squeeze(0).numpy()
    if mode != 'nearest':
        image = image.astype(np.int16)
    else:
        # print(np.unique(image))
        image = image.astype(np.byte)
    return image, new_spacing, real_resize_factor


def main(save_root):
    count = 0
    characteristics = pl.annotation_feature_names
    keys = ['sub', 'int', 'cal', 'sph', 'mar', 'lob', 'spi', 'tex', 'mal']
    scans = pl.query(pl.Scan)
    print('All scan length: ', scans.count())
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    for scan in tqdm(scans[:2]):
        pid = scan.patient_id
        mini_id = pid.split('-')[-1]
        slice_spacing = scan.slice_thickness
        pixel_spacing = scan.pixel_spacing
        spacing = np.asarray([slice_spacing, pixel_spacing, pixel_spacing])
    
        vol = scan.to_volume(verbose=False).transpose(2, 0, 1)
        res_vol, new_spacing, real_resize_factor = rescale(vol, spacing, new_spacing=[1.,1.,1.], mode='trilinear')
    
        nodes = scan.cluster_annotations()
    
        save_pkl = {}
        for nid, node in enumerate(nodes):
            diameter = np.median([anno.diameter for anno in node])
            mal = np.round(np.median([anno.malignancy for anno in node]))
            sub = np.round(np.median([anno.subtlety for anno in node]))
            ins = np.round(np.median([anno.internalStructure for anno in node]))
            sph = np.round(np.median([anno.sphericity for anno in node]))
            mar = np.round(np.median([anno.margin for anno in node]))
            lob = np.round(np.median([anno.lobulation for anno in node]))
            spi = np.round(np.median([anno.spiculation for anno in node]))
            tex = np.round(np.median([anno.texture for anno in node]))
            center = np.mean(np.asarray([anno.centroid for anno in node]), axis=0)
            center = np.asarray([center[2], center[0], center[1]])
            cal = [anno.calcification for anno in node if anno.calcification != 6]
            cal = 6 if len(cal) == 0 else np.argmax(np.bincount(cal))
    
            save_pkl['mal'] = mal
            save_pkl['sub'] = sub
            save_pkl['int'] = ins
            save_pkl['sph'] = sph
            save_pkl['mar'] = mar
            save_pkl['lob'] = lob
            save_pkl['spi'] = spi
            save_pkl['tex'] = tex
            save_pkl['cal'] = cal
            # save_pkl['diameter'] = diameter
            # save_pkl['center'] = center
            # save_pkl['rescale_diameter'] = diameter * spacing[-1]
            rescale_center = np.asarray(center) * spacing
            # save_pkl['rescale_center'] = rescale_center
    
            nodule_matrix = get_nodule_matrix(res_vol, rescale_center)
            save_pkl['matrix'] = nodule_matrix
            try:
                assert nodule_matrix.shape == (64, 128, 128)
            except:
                import pdb; pdb.set_trace()
    
            count += 1
            save_path = os.path.join(save_root, 'P{}_N{}_I{}.pkl'.format(mini_id, str(nid+1).zfill(2), str(count).zfill(4)))
            assert  not os.path.exists(save_path)
            with open(save_path, 'wb') as f:
                pickle.dump(save_pkl, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./LIDC_DATA")
    args = parser.parse_args()

    main(args.save_dir)


