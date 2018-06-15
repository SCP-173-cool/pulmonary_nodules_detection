#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../utils')

import numpy as np
import tensorflow as tf
import os
import random
import argparse
from datasets_reader import *


def parse_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create tensorflow record files \
        to make the project database')

    parser.add_argument('--dataset-path',
                        help='The direction of dataset path')

    parser.add_argument('--name', default='untitled',
                        help='The tfrecord dataset name')

    parser.add_argument('--outpath', default='/tmp/tfrecord',
                        help='The direction of tfrecord output path.')

    parser.add_argument('--train-pos-num', type=int, default=1,
                        help='The number of Positive samples augmentation')

    parser.add_argument('--train-neg-num', type=int, default=1,
                        help='The number of Negative samples augmentation')

    parser.add_argument('--box', type=int, nargs='+', default=[40, 40, 24],
                        help='The fetch box from image_arrays')

    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='The ratio of train set and valid set.')

    parser.add_argument('--image-scale', type=float, default=1.,
                        help='The scale of image datasets')
    args = parser.parse_args()
    return args


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tfrecord_string(image, label):
    feature = {
        'label': _int64_feature([label]),
        'image': _bytes_feature(image.tostring()),
        'image_shape': _int64_feature(image.shape)
    }
    tf_features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=tf_features)

    tf_serialized = example.SerializeToString()
    return tf_serialized


def train_maker(args, dataset):

    positive_augment_number = args.train_pos_num
    negative_augment_number = args.train_neg_num

    box = args.box
    scale = args.image_scale
    train_ratio = args.train_ratio
    output_path = os.path.join(args.outpath, args.name)

    train_writer = tf.python_io.TFRecordWriter(
        os.path.join(output_path, 'train.tfrecord'))
    valid_writer = tf.python_io.TFRecordWriter(
        os.path.join(output_path, 'valid.tfrecord'))
    
    print("Start making TRAIN and VALIDATION tensorflow record.")
    print("Starting ...\n")

    for train_scan_id in dataset.train_scanID_lst[:2]:
        images, resize_factor = dataset.read_images_array(
            train_scan_id, rescale=scale)
        message = dataset.read_voxel_labels(train_scan_id, resize_factor)

        mess_lst = [i for i in message if i[4] == 1] * positive_augment_number
        mess_lst += [i for i in message if i[4] == 0] * negative_augment_number

        random.shuffle(mess_lst)
        edge = len(mess_lst) * train_ratio

        for i in range(len(mess_lst)):
            nodules = nodules_reader_3D(images, mess_lst[i], box)
            nodules = np.expand_dims(nodules, axis=3)
            label = int(mess_lst[i][4])
            if nodules.shape[-2] < box[2]:
                continue
            tf_serialized = tfrecord_string(nodules, label)

            if i < edge:
                train_writer.write(tf_serialized)
            else:
                valid_writer.write(tf_serialized)
        print('{} is completed.'.format(train_scan_id))
    train_writer.close()
    valid_writer.close()

    print("The TRAIN and VALIDATION tensorflow record are finished.")


def test_maker(args, dataset):
    
    box = args.box
    scale = args.image_scale
    output_path = os.path.join(args.outpath, args.name)

    test_writer = tf.python_io.TFRecordWriter(
        os.path.join(output_path, 'test.tfrecord'))
    
    print("Start making TEST tensorflow record.")
    print("Starting ...\n")

    for test_scan_id in dataset.test_scanID_lst[:2]:
        images, resize_factor = dataset.read_images_array(
            test_scan_id, rescale=scale)
        message = dataset.read_voxel_labels(test_scan_id, resize_factor)

        for i in range(len(message)):
            nodules = nodules_reader_3D(images, message[i], box)
            nodules = np.expand_dims(nodules, axis=3)
            label = int(message[i][4])
            if nodules.shape[-2] < box[2]:
                continue
            tf_serialized = tfrecord_string(nodules, label)

            test_writer.write(tf_serialized)
        print('{} is completed.'.format(test_scan_id))
    
    test_writer.close()
    print("The TEST tensorflow record are finished.")

if __name__ == '__main__':
    args = parse_args()

    dataset = pulmonary_nodules_dataset(args.dataset_path)
    dataset.check_all()
    positive_samples_number = len(
        dataset.candidate_df[dataset.candidate_df[5] == 1])
    negative_samples_number = len(
        dataset.candidate_df[dataset.candidate_df[5] == 0])

    print("The positive samples number is {}".format(positive_samples_number))
    print("the negative samples number is {}".format(negative_samples_number))

    if not os.path.exists(args.outpath):
        os.system('mkdir {}'.format(args.outpath))
    if not os.path.exists(os.path.join(args.outpath, args.name)):
        os.system('mkdir {}'.format(os.path.join(args.outpath, args.name)))

    train_maker(args, dataset)
    test_maker(args, dataset)
