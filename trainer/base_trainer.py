#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import sys
import os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('../'))
from data_loaders.data_io import data_loader

import tensorflow as tf

class trainer(object):

    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
    
    def input_data(self):
        train_loader = data_loader()

class Save_and_load_mode(object):
    def __init__(self, logdir, sess):
        self.saver = tf.train.Saver()
        self.logdir = logdir
        self.sess = sess

    def save_model(self, step):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.saver.save(self.sess, os.path.join(
            self.logdir, 'model.ckpt'), global_step=step)

    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.logdir)
        if ckpt:
            self.saver.restore(self.sess, os.path.join(self.logdir, 'model.ckpt'))
            return True
        else:
            return False

class TensorBoard(object):
    def __init__(self):
        pass

    def variable_summaries(self,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def image_summary(self,name,tensor,max_outputs=10):
        tf.summary.image(name, tensor, max_outputs)

    def hist_summary(self,name,values):
        tf.summary.histogram(name, values)

    def scalar_summary(self,name,tensor):
        tf.summary.scalar(name, tensor)

    def merge_all_summary(self):
        return tf.summary.merge_all()

    def FileWriter_summary(self,log_dir,graph=None):
        return tf.summary.FileWriter(log_dir,graph)


