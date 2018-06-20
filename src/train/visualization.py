#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktarxiao
"""

import tensorflow as tf


class TensorBoard(object):
    def __init__(self):
        pass

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def image_summary(self, name, tensor, max_outputs=10):
        tf.summary.image(name, tensor, max_outputs)

    def hist_summary(self, name, values):
        tf.summary.histogram(name, values)

    def scalar_summary(self, name, tensor):
        tf.summary.scalar(name, tensor)

    def merge_all_summary(self):
        return tf.summary.merge_all()

    def FileWriter_summary(self, log_dir, graph=None):
        return tf.summary.FileWriter(log_dir, graph)

