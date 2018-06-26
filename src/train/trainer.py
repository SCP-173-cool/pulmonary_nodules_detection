#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktarxiao
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np

from model import MODEL
from alexnet import alexnet
from data_io import data_loader
from visualization import TensorBoard
from config import *


class base_trainer(object):
    """Trainer Module for most projects.
    - load_data
    - load_model
    - project summary
    - restore model or initialization
    - train
    - train_step
    - valid_step
    - inference
    - inference one step
    """

    def __init__(self):
        pass

    def _load_data(self):
        """Data Loader Module
        3 data loader is train, validation and test loader
        return iterator and loader.

        note: `iterator` need to be initialized before using it 
                or when the loader is out of run.
        """
        with tf.variable_scope('Data_Loader'):
            self.train_iterator = data_loader(**config_train_dataloader)
            self.valid_iterator = data_loader(**config_valid_dataloader)
            self.test_iterator = data_loader(**config_test_dataloader)

            self.train_loader = self.train_iterator.get_next()
            self.valid_loader = self.valid_iterator.get_next()
            self.test_loader = self.test_iterator.get_next()

    def _load_model(self, network):
        """Model Loader Module
        Load model from `MODEL`
        """
        with tf.variable_scope('Model'):
            self.model = MODEL(**config_model)
            self.model.build_model(network=network)

    def _summary(self):
        summary_tool = TensorBoard()
        summary_tool.scalar_summary('All_loss', self.model.all_loss)
        summary_tool.scalar_summary('negative_loss', self.model.neg_feat_loss)
        summary_tool.scalar_summary('Accuracy', self.model.accuracy)
        summary_tool.scalar_summary('Recall', self.model.recall)
        summary_tool.scalar_summary('Precision', self.model.precision)
        summary_tool.hist_summary('Predict_prob', self.model.probability)
        summary_tool.hist_summary('pos_features', self.model.pos_features)
        summary_tool.hist_summary('neg_features', self.model.neg_features)
        summary_tool.hist_summary('pos_probability', self.model.pos_prob)
        summary_tool.hist_summary('neg_probability', self.model.neg_prob)
        
        summary_tool.image_summary(
            'pos_image', self.model.pos_image, max_outputs=2)
        summary_tool.image_summary(
            'neg_image', self.model.neg_image, max_outputs=2)

        self.summary_ops = summary_tool.merge_all_summary()

    def _initialization(self, sess):
        model_file = tf.train.latest_checkpoint(self.save_path)
        try:
            self.saver.restore(sess, model_file)
            print('Restore Sucessful!')
        except:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print('Restore Failed!')

    def train(self, sess):
        # load data and model
        self._load_data()
        self._load_model(network=alexnet)
        self._summary()

        # set the saver path
        self.save_path = os.path.join(
            config_saver['train_results'], config_saver['save_name'])

        # building Trainer
        with tf.variable_scope('Trainer'):

            self.lr_placeholder = tf.placeholder_with_default(
                0.1, shape=[], name='learning_rate')
            self.tvars = tf.trainable_variables()

            self.grads = tf.gradients(self.model.all_loss, self.tvars)
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.lr_placeholder, momentum=0.9)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.grads, self.tvars))

        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), max_to_keep=1)
        self.writer = tf.summary.FileWriter(os.path.join(self.save_path, 'train_summary'), sess.graph)

        print("Trainable Parameters:")
        for tvar in self.tvars:
            print(tvar.name)
        print("=================================================================")

        self._initialization(sess)
        sess.run(self.train_iterator.initializer)

        self.global_step = 0
        self.mini_accuracy = 0

        for epoch in range(config_train['epochs']):
            self.epoch = epoch
            self.train_step(sess)

        self.writer.close()

    def train_step(self, sess):
        accuracy_lst = []
        precision_lst = []
        recall_lst = []
        for step in range(config_train['steps_per_epoch']):
            try:
                image_batch, label_batch = sess.run([self.train_loader])[0]
            except tf.errors.OutOfRangeError:
                sess.run(self.train_iterator.initializer)

            train_feed = {self.model.image: image_batch,
                          self.model.label: label_batch,
                          self.lr_placeholder: 0.001}
            run_lst = [self.train_op, self.summary_ops, self.model.likelihood_loss,
                       self.model.neg_feat_loss, self.model.accuracy,
                       self.model.recall, self.model.precision]
            _, summary, loss, neg_loss, acc, recall, precision = sess.run(
                run_lst, feed_dict=train_feed)
            accuracy_lst.append(acc)
            recall_lst.append(recall)
            precision_lst.append(precision)
            self.writer.add_summary(summary, self.global_step)
            print("[epoch: {}]: loss: {:.4f}, neg_loss:{:.4f}, accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}".format(
                self.epoch, loss, neg_loss, np.mean(accuracy_lst), np.mean(precision_lst), np.mean(recall_lst)))

            if self.global_step % config_valid['validation_loop'] == 0 and self.global_step > 0:
                valid_accuracy = self.valid_step(sess)
                if valid_accuracy > self.mini_accuracy:
                    self.mini_accuracy = valid_accuracy
                    self.saver.save(sess, os.path.join('model_save', 'model'))

            self.global_step += 1

    def valid_step(self, sess):
        result_lst = []
        sess.run(self.valid_iterator.initializer)
        while True:
            try:
                image_batch, label_batch = sess.run([self.valid_loader])[0]
            except tf.errors.OutOfRangeError:
                break
            valid_feed = {self.model.image: image_batch,
                          self.model.label: label_batch}
            run_lst = [self.model.likelihood_loss, self.model.neg_feat_loss,
                       self.model.accuracy, self.model.recall,
                       self.model.precision]
            result_lst.append(sess.run(run_lst, feed_dict=valid_feed))
        results = list(np.array(result_lst).sum(axis=0)/len(result_lst))

        print("\nvalidatioin process")
        print("Loss:{}, negative_loss:{}, Accuracy:{}, Recall:{}, Precision:{}".format(
            *results))

        return results[2]
