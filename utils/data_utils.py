#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""
import sys
sys.dont_write_bytecode = True


import warnings
import cv2
import os

import SimpleITK as sitk
import pydicom as dcm
import numpy as np
from scipy.ndimage.interpolation import zoom


def read_dicom(dicom_dir):
    """
    read a series of dicom from 'dicom_dir'
    Args:
        dicom_dir: a directory that contains a series of '.dcm' files
    Ret:
        SimpleITK.Image
    """
    reader = sitk.ImageSeriesReader()
    dicom_file_names = reader.GetGDCMSeriesFileNames(dicom_dir)

    n_dicom_file = len([file_name for file_name in os.listdir(
        dicom_dir) if file_name.endswith('.dcm')])
    assert n_dicom_file == len(dicom_file_names)

    reader.SetFileNames(dicom_file_names)
    dicom = reader.Execute()

    return dicom


def read_dicom_meta(dicom_dir):
    """
    read meta information in 'dicom_dir'
    Args:
        dicom_dir: a directory that contains a series of '.dcm' files
    Rets:
        a list of meta information of each slice of dicom in 'dicom_dir'
    """
    reader = sitk.ImageSeriesReader()
    dicom_file_names = reader.GetGDCMSeriesFileNames(dicom_dir)

    infos = list()
    for dicom_file_name in dicom_file_names:
        dataset = dcm.read_file(dicom_file_name)
        del dataset[0x7fe0, 0x0010]
        infos.append(dataset)

    return list(reversed(infos))


def get_ndarray_from_dicom(dicom):
    """
    Get numpy array from a dicom class which is read by function 'read_dicom'
    Args:
        dicom: SimpleITK.Image class which is read by function 'read_dicom'
    Rets:
        the numpy array from dicom images, whose values are STILL 'HU', and the type is 'np.float32'
    """
    ndarray = sitk.GetArrayFromImage(dicom)
    ndarray = np.flip(ndarray, 0)
    ndarray = np.rollaxis(ndarray, axis=0, start=3)
    ndarray[np.where(ndarray < -1024.)] = -1024.
    return ndarray.astype(np.float32)


def read_mhd_scan(mhd_path):
    """
    Get all messages from a '.mhd' file
    Args:
        mhd_path: a file direction that is ended with '.mhd'
    Rets:
        image_arrays: the numpy array from mhd file, whose values are STILL 'HU'.
        origin: the array of images origin
        spacing: the array of images spacing
        size: the shape of image_arrays
    """
    scan = sitk.ReadImage(mhd_path)
    image_arrays = sitk.GetArrayFromImage(scan)
    image_arrays = np.rollaxis(image_arrays, axis=0, start=3)
    image_arrays = image_arrays[:, :, ::-1]
    origin = np.asarray(scan.GetOrigin())

    spacing = np.asarray(scan.GetSpacing())
    return image_arrays, origin, spacing


def read_CT_image(path, resolution='keep'):
    if path.endswith(".mhd"):
        image_arrays, _, _ = read_mhd_scan(path)
    else:
        image_arrays = get_ndarray_from_dicom(read_dicom(path))
    image_arrays = transform_voxel_range(image_arrays, 0, 255)
    return image_arrays.astype(np.uint8)


def transform_voxel_range(image_arrays, new_lower, new_upper, original_lower=-1000, original_upper=400.):
    """
    First squeeze the range of element in 'image_arrays' to  [original_lower, original_upper],
    Then convert to new range [new_lower, new_upper]
    """
    image_arrays[np.where(image_arrays < original_lower)] = original_lower
    image_arrays[np.where(image_arrays > original_upper)] = original_upper
    image_arrays_of_new_range = (image_arrays - original_lower) / (
        original_upper - original_lower) * (new_upper - new_lower) + new_lower

    return image_arrays_of_new_range


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def resample(imgs, spacing, new_spacing):

    new_shape = np.round(imgs.shape * spacing / new_spacing)
    resize_factor = new_shape / imgs.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imgs = zoom(imgs, resize_factor, mode='nearest')
    return imgs, resize_factor


def label_message_image_series(images, message, color=0):
    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), ]
    color = color_list[color]
    save_lst = []
    images = images.copy()
    for mess in message:
        lst = [mess[2]]
        lst = [int(round(tt)) for tt in lst]
        x = int(round(mess[0] - 22/2))
        y = int(round(mess[1] - 22/2))
        wh = 22
        for z in lst:
            cv2.rectangle(images[z], (x, y), (x+wh, y+wh), color, 1)
        save_lst += lst

    return images, save_lst


if __name__ == '__main__':
    mhdpath = '/data/pulmonary_nodules/LUNA/Data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.219087313261026510628926082729.mhd'
    image_arrays, origin, spacing = read_mhd_scan(mhdpath)

    new_spacing = np.array([1, 1, 1])
    newimg, resize_factor = resample(
        image_arrays, spacing, new_spacing, order=1)
    newimg = transform_voxel_range(newimg, 0, 255).astype(np.uint8)

    import pandas as pd
    df = pd.read_csv(
        '/data/pulmonary_nodules/LUNA/Meta/LUNA_voxel_v3.csv', header=None)
    mess = df[df[0] == 'Data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.219087313261026510628926082729.mhd'][[1,
                                                                                                             2, 3]].values.astype(np.int32)*resize_factor

    from skimage import io
    newimg = np.stack([newimg, newimg, newimg], axis=2)
    newimg = np.transpose(newimg, [3, 0, 1, 2])
    images, save_lst = label_message_image_series(newimg, mess, 1)
    from tqdm import tqdm
    for z in tqdm(save_lst):
        io.imsave('output/{}.png'.format(str(z)), images[z])
