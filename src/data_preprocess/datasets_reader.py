#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True  # will not create *.pyc file

import pydicom as dcm
import SimpleITK as sitk

import numpy as np
import os
import time
import warnings
import cv2
import pandas as pd

from scipy.ndimage.interpolation import zoom


def _check_filepath(filepath, name, raiseout=False, printout=True):
    """check if any file exists and print message of the file.

    Args:
        filepath: string, the path of file which you want to check if exists.
        name: string, file name which you want print out.
        raiseout: bool, whether raise Error and break out when doesn't exists, default False.
        printout: bool, control the printed messages, default True.

    Rets:
        _: bool, if exists the file in `filepath`

    """
    if os.path.exists(filepath):
        if printout:
            print('[PASS] The {} path is {}'.format(name, filepath))
        return True
    else:
        if printout:
            print('[ERROR] Does not exist {} path: {}'.format(name, filepath))
        if raiseout:
            raise RuntimeError('does not exist this path')
        return False


class pulmonary_nodules_dataset(object):
    """Tool of reading pulmonary nodules standard dataset
    """

    def __init__(self, dataset_path, ):
        self.root_path = dataset_path
        self.data_path = os.path.join(self.root_path, 'Data')
        self.meta_path = os.path.join(self.root_path, 'Meta')

        self.annotation_file = os.path.join(self.meta_path, 'nodules')
        self.index_file = os.path.join(self.meta_path, 'index')
        self.candidate_csv = os.path.join(self.meta_path, 'voxel.csv')
        self.testset_file = os.path.join(self.meta_path, 'testset')

        if _check_filepath(self.annotation_file, 'Annotation', printout=False):
            self.annotation_df = pd.read_csv(
                self.annotation_file, header=None, sep='\t')
        else:
            self.annotation_df = None

        if _check_filepath(self.index_file, 'Index File', printout=False):
            self.index_df = pd.read_csv(
                self.index_file, header=None, sep='\t')
        else:
            self.index_df = None

        if _check_filepath(self.candidate_csv, 'Candidate File', printout=False):
            self.candidate_df = pd.read_csv(self.candidate_csv, header=None)
            self.all_scanID_lst = list(set(self.candidate_df[0].values))
        else:
            self.candidate_df = None
            self.all_scanID_lst = []

        if _check_filepath(self.testset_file, 'TestSet File', printout=False):
            self.test_scanID_lst = list(
                set(pd.read_csv(self.testset_file, header=None, sep='\t')[0].values))
            self.train_scanID_lst = [
                scan_id for scan_id in self.all_scanID_lst if scan_id not in self.test_scanID_lst]

        else:
            self.test_scanID_lst = []
            self.train_scanID_lst = self.all_scanID_lst

    def check_all(self):

        print('***************************')
        print('Begin to check datasets...\n')

        # direction check
        print('Checking directions...')
        time.sleep(0.5)
        _check_filepath(self.root_path, 'DataSet folder', raiseout=True)
        _check_filepath(self.data_path, 'Data Keeping', raiseout=True)
        _check_filepath(self.meta_path, 'Data Meta', raiseout=True)
        _check_filepath(self.annotation_file, 'Annotation File')
        _check_filepath(self.candidate_csv, 'Candidate.csv File')
        _check_filepath(self.testset_file, 'testset File')
        _check_filepath(self.index_file, 'Index File')

        # file check
        print('Checking data files...')
        time.sleep(0.5)

        # Train set processing
        self.num_train_neg = 0
        self.num_train_pos = 0
        del_lst = []
        for scan_id in self.train_scanID_lst:
            filepath = os.path.join(self.root_path, scan_id)
            res = _check_filepath(filepath, 'nodules file', printout=False)
            if not res:
                del_lst.append(scan_id)
            else:
                tmp_df = self.candidate_df[self.candidate_df[0] == scan_id]
                self.num_train_neg += len(tmp_df[tmp_df[5]==0])
                self.num_train_pos += len(tmp_df[tmp_df[5]==1])

        self.train_scanID_lst = [
            i for i in self.train_scanID_lst if i not in del_lst]
        self.all_scanID_lst = [
            i for i in self.all_scanID_lst if i not in del_lst]

        for scan_id in del_lst:
            self.candidate_df = self.candidate_df[~(
                self.candidate_df[0] == scan_id)]

        # Test set processing
        self.num_test_neg = 0
        self.num_test_pos = 0
        del_lst = []
        for scan_id in self.test_scanID_lst:
            filepath = os.path.join(self.root_path, scan_id)
            res = _check_filepath(filepath, 'nodules file', printout=False)
            if not res:
                del_lst.append(scan_id)
            else:
                tmp_df = self.candidate_df[self.candidate_df[0] == scan_id]
                self.num_test_neg += len(tmp_df[tmp_df[5]==0])
                self.num_test_pos += len(tmp_df[tmp_df[5]==1])

        self.test_scanID_lst = [
            i for i in self.test_scanID_lst if i not in del_lst]
        self.all_scanID_lst = [
            i for i in self.all_scanID_lst if i not in del_lst]

        for scan_id in del_lst:
            self.candidate_df = self.candidate_df[~(
                self.candidate_df[0] == scan_id)]
    
        
        print('\nAll checks finished !')
        print('The number of trainset positive samples is {}'.format(self.num_train_pos))
        print('The number of trainset negative samples is {}'.format(self.num_train_neg))
        print('The number of testset positive samples is {}'.format(self.num_test_pos))
        print('The number of testset negative samples is {}'.format(self.num_test_neg))

    def read_images_meta(self, scan_id):
        message_df = self.index_df[self.index_df[0] == scan_id]
        assert len(message_df) == 1
        images_shape = np.array(
            message_df[[2, 3, 4]].values[0], dtype=np.int32)
        images_spacing = np.array(
            message_df[[5, 6, 7]].values[0], dtype=np.float32)
        return images_shape, images_spacing

    def read_images_array(self, scan_id, rescale=-1):
        readpath = os.path.join(self.root_path, scan_id)
        images = read_CT_image(readpath)
        images_shape, images_spacing = self.read_images_meta(scan_id)
        assert tuple(images_shape) == images.shape

        if rescale == -1:
            new_spacing = images_spacing
        elif rescale > 0:
            new_spacing = np.ones(3) * images_spacing[0] * 1.0 / rescale
        else:
            raise Exception

        images_array, resize_factor = resample(
            images, images_spacing, new_spacing)

        images_array = transform_voxel_range(
            images_array, 0, 255).astype(np.uint8)

        return images_array, resize_factor

    def read_voxel_labels(self, scan_id, resize_factor=[1, 1, 1]):
        voxel_labels_df = self.candidate_df[self.candidate_df[0] == scan_id]
        labels_message = np.array(voxel_labels_df[[1, 2, 3, 4, 5, 6]].values)
        labels_message[:, :3] *= resize_factor
        labels_message[:, 3] *= resize_factor[0]
        return labels_message

    def read_GT_labels(self, scan_id, resize_factor=[1, 1, 1]):
        GT_labels_df = self.annotation_df[self.annotation_df[0] == scan_id]
        labels_message = np.array(GT_labels_df[[2, 3, 4, 5]].values)
        labels_message[:, :3] *= resize_factor
        labels_message[:, 3] *= resize_factor[0]
        return labels_message


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


def read_CT_image(path):
    if path.endswith(".mhd"):
        image_arrays, _, _ = read_mhd_scan(path)
    else:
        image_arrays = get_ndarray_from_dicom(read_dicom(path))
    return image_arrays


def nodules_reader_2D(images_array, coord, r=22):
    x = int(round(coord[1]))
    y = int(round(coord[0]))
    z = int(round(coord[2]))

    if images_array.shape[-1] == 3 and images_array.ndim == 4:
        nodules = images_array[z, x - r:x + r, y - r:y + r, :]
    elif images_array.ndim == 3:
        nodules = images_array[x - r:x + r, y - r:y + r, z]
    else:
        print(
            "Wrong type! The image shape should be [z, x, y ,3] or [x, y, z]")
        return None
    return nodules


def nodules_reader_3D(images_array, coord, box=[22, 22, 22]):
    x = int(round(coord[1]))
    y = int(round(coord[0]))
    z = int(round(coord[2]))
    box_size = box
    box = np.array(box_size) / 2
    box = box.astype(np.int64)

    x_1 = max(0, x - box[0])
    x_2 = min(images_array.shape[0], x + box[0]+1)
    y_1 = max(0, y - box[1])
    y_2 = min(images_array.shape[1], y + box[1]+1)
    z_1 = max(0, z - box[2])
    z_2 = min(images_array.shape[2], z + box[2]+1)

    padding = np.zeros(box_size)
    if images_array.ndim == 3:
        nodules = images_array[x_1:x_2, y_1:y_2, z_1:z_2]
        padding[0:nodules.shape[0], 0:nodules.shape[1], 0:nodules.shape[2]] = nodules
        nodules = padding.copy()
    else:
        print("Wrong type! The image shape should be [x, y, z]")
        return None
    return nodules


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


def label_message_image_series(images, message, color=0, z_range=0):
    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), ]
    color = color_list[color]
    save_lst = []
    images = images.copy()
    for mess in message:
        z = int(round(mess[2]))
        lst = list(range(max(0, z - z_range),
                         min(images.shape[2], z + z_range)))
        wh = int(round(mess[3]))
        x = int(round(mess[0] - wh / 2))
        y = int(round(mess[1] - wh / 2))

        for z in lst:
            cv2.rectangle(images[z], (x, y), (x + wh, y + wh), color, 1)
        save_lst += lst

    return images, save_lst


def images2tif(images, save_dir, duration=0.05):
    try:
        import imageio
    except:
        print("Maybe You can try : 'pip install -U imageio'")

    if isinstance(images, np.ndarray) and images.ndim >= 3:
        images_lst = [images[i] for i in range(images.shape[0])]
        imageio.mimsave(save_dir, images_lst, duration=duration)
    elif isinstance(images, list):
        imageio.mimsave(save_dir, images, duration=duration)

    else:
        print("Wrong images type which is not list type or numpy.array")
        raise Exception


if __name__ == "__main__":
    dataset = pulmonary_nodules_dataset('/data/pulmonary_nodules/LUNA')
    dataset.check_all()
    import pdb; pdb.set_trace()
    """
    scan_id = dataset.all_scanID_lst[0]
    images, resize_factor = dataset.read_images_array(scan_id, rescale=1)
    message = dataset.read_voxel_labels(scan_id, resize_factor)
    gt_message = dataset.read_GT_labels(scan_id, resize_factor)
    """

    """
    images_new = np.transpose(
        np.stack([images, images, images], axis=2), [3, 0, 1, 2])
    images_new, save_lst1 = label_message_image_series(
        images_new, message, color=1, z_range=3)
    images_new, save_lst2 = label_message_image_series(
        images_new, gt_message, color=2, z_range=2)
    save_lst = list(set(save_lst1+save_lst2))

    from tqdm import tqdm
    from skimage import io
    for z in tqdm(save_lst):
        io.imsave('output/{}.png'.format(str(z)), images_new[z])
    """
    # images2tif(images_new, 'new.gif', duration=0.1)

    #nodules = nodules_reader_3D(images, message[0], box=[22, 22, 22])
