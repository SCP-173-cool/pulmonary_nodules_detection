#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import sys
sys.dont_write_bytecode = True
import tensorflow as tf
import numpy as np


class MODEL(object):

    def __init__(self, 
                input_shape, 
                num_channel):
        self.input_shape = input_shape
        self.num_channel = num_channel
        self.image_shape = [None]+self.input_shape+[self.num_channel]

    def build_model(self, network, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("input_block", reuse=reuse):
            
            self.image = tf.placeholder(tf.float32, shape=self.image_shape, name='image')
            self.label = tf.placeholder(tf.float32, shape=None, name='label')

        with tf.variable_scope("feature", reuse=reuse):
            self.feature = network(self.image)

        with tf.variable_scope("output_block", reuse=reuse):
            avgpool = tf.layers.average_pooling3d(
                self.feature, self.feature.get_shape().as_list()[1: 4], 1)
            self.output = tf.reshape(
                avgpool, shape=[-1, int(np.prod(avgpool.get_shape()[1:]))])

            self.y_ = tf.layers.dense(self.output, 1)
            self.y_ = tf.squeeze(self.y_, axis=1)
            self.probability = tf.nn.sigmoid(self.y_)

        with tf.variable_scope("loss", reuse=reuse):
            self.negative_feature_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(self.output), axis=1) * (1 - self.label))
            self.likelihood_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.label, logits=self.y_)
            self.likelihood_loss = tf.reduce_mean(self.likelihood_loss)

            self.all_loss = self.likelihood_loss + self.negative_feature_loss



if __name__ == '__main__':
    sys.path.insert(0, '../data_loaders')
    from data_io import data_loader
    from alexnet import alexnet

    train_tfrecord_lst = ['../output/LUNA/train.tfrecord']
    train_iterator = data_loader(train_tfrecord_lst,
                               num_repeat=1,
                               shuffle=True,
                               batch_size=2,
                               num_processors=4,
                               augmentation=True,
                               name='train_dataloader')
    model = MODEL([36, 36, 20], 1)
    model.build_model(network=alexnet, reuse=tf.AUTO_REUSE)

    train_loader = train_iterator.get_next()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)

    for i in range(22):
        try:
            image, label = sess.run([train_loader])[0]
            feed_dict = {model.image: image, model.label: label}
            loss, prob = sess.run([model.all_loss, model.probability], feed_dict=feed_dict)
            print(loss, prob)
        except tf.errors.OutOfRangeError:
            break
