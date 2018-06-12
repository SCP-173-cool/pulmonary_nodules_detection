#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import sys
sys.dont_write_bytecode = True

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from models.model import MODEL
from models.alexnet import alexnet
from data_loaders.data_io import data_loader

import tensorflow as tf

from config import *


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

class base_trainer(object):
    def __init__(self):
        pass
    def _load_data(self):
        with tf.variable_scope('Data_Loader'):
            self.train_iterator = data_loader(**config_train_dataloader)
            self.valid_iterator = data_loader(**config_valid_dataloader)
            self.test_iterator  = data_loader(**config_test_dataloader)

            self.train_loader = self.train_iterator.get_next()
            self.valid_loader = self.valid_iterator.get_next()
            self.test_loader  = self.test_iterator.get_next()

    def _load_model(self, network):
        with tf.variable_scope('Model'):
            self.model = MODEL(**config_model)
            self.model.build_model(network=network)

    def _summary(self):
        summary_tool = TensorBoard()
        summary_tool.scalar_summary('All_loss', self.model.all_loss)
        summary_tool.scalar_summary('negative_loss', self.model.negative_feature_loss)
        summary_tool.hist_summary('Predict_prob', self.model.probability)
        summary_tool.image_summary('input_image', self.model.image[:,:,:,9:12,0])
        
        self.summary_ops = summary_tool.merge_all_summary()

    def train(self, sess):
        self._load_data()
        self._load_model(network=alexnet)
        self._summary()
        with tf.variable_scope('trainer'):
            self.tvars = tf.trainable_variables()
            self.lr_placeholder = tf.placeholder_with_default(0.01, shape=[], name='learning_rate')
            self.grads = tf.gradients(self.model.all_loss, self.tvars)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
            self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.tvars))
        
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        self.writer = tf.summary.FileWriter('log', sess.graph)

        print("Trainable Parameters:")
        for tvar in self.tvars:
            print(tvar.name)
        print("=================================================================")

        sess.run(tf.global_variables_initializer())
        sess.run(self.train_iterator.initializer)

        self.global_step = 0
        for epoch in range(3):
            self.epoch = epoch
            self.train_step(sess)

        self.writer.close()

    def train_step(self, sess):
        for step in range(config_train['steps_per_epoch']):
            try:
                image_batch, label_batch = sess.run([self.train_loader])[0]
            except tf.errors.OutOfRangeError:
                sess.run(self.train_iterator.initializer)

            train_feed = {self.model.image: image_batch, 
                         self.model.label: label_batch,
                         self.lr_placeholder: 0.001}
            _, summary, loss = sess.run([self.train_op, 
                                         self.summary_ops, 
                                         self.model.all_loss], 
                                         feed_dict=train_feed)
            self.writer.add_summary(summary, self.global_step)
            print("epoch {}: loss is {}".format(self.epoch, loss), end='\r')


            self.global_step += 1






if __name__ == "__main__":
    trainer = base_trainer()
    sess = tf.Session()
    trainer.train(sess)
    sess.close()


            



