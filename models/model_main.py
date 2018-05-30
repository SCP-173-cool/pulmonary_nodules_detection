#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../data_loaders')

import tensorflow as tf
from alexnet import alexnet
import numpy as np

from data_io import data_loader


class MODEL(object):

    def __init__(self, 
                input_shape, 
                num_channel,
                data_loader=None):
        self.input_shape = input_shape
        self.num_channel = num_channel
        self.image_shape = [None]+self.input_shape+[self.num_channel]
        self.data_loader = data_loader

    def build_model(self):
        with tf.variable_scope("input_block"):
            
            if self.data_loader is None:
                self.image = tf.placeholder(tf.float32,
                                            shape=self.image_shape, name='image1')

                self.label = tf.placeholder(tf.float32, shape=None, name='label')
            
            else:
                self.image, self.label = self.data_loader
        

        with tf.variable_scope("feature"):
            self.feature = alexnet(self.image)

        with tf.variable_scope("output_block"):
            avgpool = tf.layers.average_pooling3d(
                self.feature, self.feature.get_shape().as_list()[1: 4], 1)
            self.output = tf.reshape(
                avgpool, shape=[-1, int(np.prod(avgpool.get_shape()[1:]))])

            self.y_ = tf.layers.dense(self.output, 1)
            self.y_ = tf.squeeze(self.y_, axis=1)
            self.probability = tf.nn.sigmoid(self.y_)

        with tf.variable_scope("loss"):
            self.negative_feature_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(self.output), axis=1) * (1 - self.label))
            self.likelihood_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.label, logits=self.y_)

            self.loss = self.likelihood_loss + self.negative_feature_loss

        with tf.variable_scope("control_params"):
            self.init = tf.global_variables_initializer()


if __name__ == '__main__':


    train_tfrecord_lst = ['../output/LUNA/train.tfrecord']
    train_loader = data_loader(train_tfrecord_lst,
                               num_repeat=1,
                               shuffle=True,
                               batch_size=1,
                               num_processors=4,
                               augmentation=True,
                               name='train_dataloader')
    model = MODEL([36, 36, 20], 1, data_loader=train_loader)
    model.build_model()

    sess = tf.Session()
    sess.run(model.init)

    for i in range(22):
        try:
            #image, label = sess.run([train_loader])[0]
            loss, prob = sess.run([model.loss, model.probability])
            print(loss, prob)
        except tf.errors.OutOfRangeError:
            break
