#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import tensorflow as tf


def _conv3d(input_tensor, ksize, num_filter, name):
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.nn.l2_loss
    activition = tf.nn.leaky_relu
    activity_regularizer = tf.layers.batch_normalization

    output = tf.layers.conv3d(input_tensor, num_filter, ksize,
                              padding='SAME',
                              activation=activition,
                              activity_regularizer=activity_regularizer,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name=name)
    return output


def _maxpool3d(input_tensor, pool_size, stride, name):
    output = tf.layers.max_pooling3d(input_tensor, pool_size,
                                     strides=stride, name=name)
    return output


def _dropout(input_tensor, drop_rate):
    output = tf.layers.dropout(input_tensor, rate=drop_rate, name='dropout')
    return output


def _conv_block(input_tensor, num_filter, block_name, drop_rate=0, reuse=None):
    with tf.variable_scope(block_name, reuse=reuse):
        conv1 = _conv3d(input_tensor, 3, num_filter, 'conv1')
        conv2 = _conv3d(conv1, 3, num_filter, 'conv2')
        maxpool = _maxpool3d(conv2, 2, 2, 'maxpool')
        output = _dropout(maxpool, drop_rate)
    return output


def alexnet(input_tensor, reuse=None):
    with tf.variable_scope("alexnet", reuse=reuse):
        block1 = _conv_block(input_tensor, 32, "conv_block1")
        block2 = _conv_block(block1, 64, "conv_block2")
        block3 = _conv_block(block2, 128, "conv_block3")

    return block3
