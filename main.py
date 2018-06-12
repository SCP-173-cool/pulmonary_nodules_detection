#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""
from __future__ import print_function
import sys
sys.dont_write_bytecode = True

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from models.model import MODEL
from models.alexnet import alexnet
from data_loaders.data_io import data_loader

import tensorflow as tf
import numpy as np

from config import *


class TensorBoard(object):
    def __init__(self):
        pass

    def variable_summaries(self, var):
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


class base_trainer(object):
    def __init__(self):
        pass

    def _load_data(self):
        with tf.variable_scope('Data_Loader'):
            self.train_iterator = data_loader(**config_train_dataloader)
            self.valid_iterator = data_loader(**config_valid_dataloader)
            self.test_iterator = data_loader(**config_test_dataloader)

            self.train_loader = self.train_iterator.get_next()
            self.valid_loader = self.valid_iterator.get_next()
            self.test_loader = self.test_iterator.get_next()

    def _load_model(self, network):
        with tf.variable_scope('Model'):
            self.model = MODEL(**config_model)
            self.model.build_model(network=network)

    def _summary(self):
        summary_tool = TensorBoard()
        summary_tool.scalar_summary('All_loss', self.model.all_loss)
        summary_tool.scalar_summary(
            'negative_loss', self.model.negative_feature_loss)
        summary_tool.scalar_summary('Accuracy', self.model.accuracy)
        summary_tool.scalar_summary('AUC', self.model.auc)
        summary_tool.scalar_summary('Recall', self.model.recall)
        summary_tool.scalar_summary('Precision', self.model.precision)
        summary_tool.hist_summary('Predict_prob', self.model.probability)
        summary_tool.hist_summary('pos_features', self.model.pos_features)
        summary_tool.hist_summary('neg_features', self.model.neg_features)
        summary_tool.image_summary('pos_image', self.model.pos_image)
        summary_tool.image_summary('neg_image', self.model.neg_image)

        self.summary_ops = summary_tool.merge_all_summary()

    def train(self, sess):
        self._load_data()
        self._load_model(network=alexnet)
        self._summary()
        with tf.variable_scope('trainer'):
            self.tvars = tf.trainable_variables()
            self.lr_placeholder = tf.placeholder_with_default(
                0.01, shape=[], name='learning_rate')
            self.grads = tf.gradients(self.model.all_loss, self.tvars)
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.lr_placeholder, momentum=0.9)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.grads, self.tvars))

        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        self.writer = tf.summary.FileWriter('log', sess.graph)

        print("Trainable Parameters:")
        for tvar in self.tvars:
            print(tvar.name)
        print("=================================================================")

        sess.run(tf.global_variables_initializer())
        sess.run(self.train_iterator.initializer)

        self.global_step = 0
        self.mini_accuracy = 0

        for epoch in range(len(config_train['epochs'])):
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
            run_lst = [self.train_op, self.summary_ops, self.model.likelihood_loss,
                       self.model.negative_feature_loss, self.model.accuracy,
                       self.model.auc, self.model.recall, self.model.precision]
            _, summary, loss, neg_loss, acc, auc, recall, precision = sess.run(
                run_lst, feed_dict=train_feed)
            self.writer.add_summary(summary, self.global_step)
            print("\rloss: {}, neg_loss:{}, accuracy:{}, AUC:{}, precision:{}, recall:{}".format(
                self.epoch, loss, acc, auc, precision, recall), end='\r')

            if self.global_step % config_valid['validation_loop'] and self.global_step > 0:
                valid_accuracy = self.valid_step(sess)
                if valid_accuracy > self.mini_accuracy:
                    self.mini_accuracy = valid_accuracy
                    self.saver.save(sess, os.path.join('model_save','model'))

            self.global_step += 1

    def valid_step(self, sess):
        result_lst = []
        while True:
            try:
                image_batch, label_batch = sess.run([self.valid_loader])[0]
            except tf.errors.OutOfRangeError:
                break
            valid_feed = {self.model.image: image_batch,
                          self.model.label: label_batch}
            run_lst = [self.model.likelihood_loss, self.model.negative_feature_loss, 
                       self.model.accuracy, self.model.auc, self.model.recall, 
                       self.model.precision]
            result_lst.append(sess.run(run_lst, feed_dict=valid_feed))
        results = list(np.array(result_lst).sum(axis=0)/len(result_lst))

        print("\nvalidatioin process")
        print("Loss:{}, negative_loss:{}, Accuracy:{}, AUC:{}, Recall:{}, Precision:{}".format(*results))
        
        return results[2]


if __name__ == "__main__":
    trainer = base_trainer()
    sess = tf.Session()
    trainer.train(sess)
    sess.close()
