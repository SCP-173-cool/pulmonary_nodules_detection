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

    def __init__(self, input_shape, num_channel):
        self.input_shape = input_shape
        self.num_channel = num_channel
        self.image_shape = [None]+self.input_shape+[self.num_channel]

    def build_model(self):
        with tf.variable_scope("input_block"):
            self.image = tf.placeholder(tf.float32,
                                   shape=self.image_shape, name='image1')

            self.label = tf.placeholder(tf.float32, shape=None, name='label')

        with tf.variable_scope("feature"):
            self.feature = alexnet(self.image)

        with tf.variable_scope("output_block"):
            avgpool = tf.layers.average_pooling3d(
                self.feature, self.feature.get_shape().as_list()[1: 4], 1)
            self.output = tf.reshape(avgpool, shape = [-1, int(np.prod(avgpool.get_shape()[1 :]))])

        with tf.variable_scope("loss"):
            

        with tf.variable_scope("control_params"):
            self.init = tf.global_variables_initializer()


if __name__ =='__main__':
    model = MODEL([36, 36, 20], 1)
    model.build_model()

    train_tfrecord_lst = ['/tmp/tfrecord/LUNA/train.tfrecord']
    train_loader = data_loader(train_tfrecord_lst,
                               num_repeat=1,
                               shuffle=True,
                               batch_size=12,
                               num_processors=4,
                               augmentation=True,
                               name='train_dataloader')

    sess = tf.Session()
    sess.run(model.init)

    for i in range(22):
        try:
            image, label = sess.run([train_loader])[0]
            output = sess.run([model.output], feed_dict={model.image: image})[0]
            print(output.shape)
        except tf.errors.OutOfRangeError:
            break
    writer = tf.summary.FileWriter('test', sess.graph)
    sess.close()
    writer.close()
