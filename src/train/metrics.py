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

def binary_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, 
                                  tf.round(y_pred)), tf.float32), axis=-1)

def categorical_accuracy(y_true, y_pred):
    return tf.cast(tf.equal(tf.argmax(y_true, axis=-1),
                          tf.argmax(y_pred, axis=-1)), tf.float32)

def sparse_categorical_accuracy(y_true, y_pred):
    return tf.cast(tf.equal(tf.reduce_max(y_true, axis=-1),
                          tf.cast(tf.argmax(y_pred, axis=-1), tf.float32), tf.float32))