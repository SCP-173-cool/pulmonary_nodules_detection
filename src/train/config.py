#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""
import os

config_train_dataloader = {'tfrecord_lst': [os.path.abspath('../../output/tfrecords/LUNA/train.tfrecord')],
                           'num_repeat': 1, 'shuffle': True,
                           'batch_size': 6, 'num_processors': 4,
                           'augmentation': True, 'name': 'Train_Data_Loader',
                           'device': 'cpu:0'}

config_valid_dataloader = {'tfrecord_lst': [os.path.abspath('../../output/tfrecords/LUNA/valid.tfrecord')],
                           'num_repeat': 1, 'shuffle': True,
                           'batch_size': 1, 'num_processors': 4,
                           'augmentation': True, 'name': 'Valid_Data_Loader',
                           'device': 'cpu:0'}

config_test_dataloader = {'tfrecord_lst': [os.path.abspath('../../output/tfrecords/LUNA/test.tfrecord')],
                          'num_repeat': 1, 'shuffle': True,
                          'batch_size': 1, 'num_processors': 4,
                          'augmentation': True, 'name': 'Test_Data_Loader',
                          'device': 'cpu:0'}

config_model = {'input_shape': [36, 36, 20], 'num_channel': 1}

config_train = {'steps_per_epoch': 1000, 'epochs': 10, 'learning_rate': 0.001}
config_valid = {'validation_loop': 100}

config_saver = {'train_results': os.path.abspath('../../output/train_results'),
                'save_name': 'alexnet'}
