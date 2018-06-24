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

def epsilon():
    return tf.constant(sys.float_info.epsilon, tf.float32)

def _True_Positive(y_true, y_pred):
    with tf.variable_scope('True_Positive'):
        predicted = tf.round(y_pred)
        actual = y_true
        TP = tf.count_nonzero(predicted * actual)
    return tf.cast(TP, tf.float32)

def _True_Negative(y_true, y_pred):
    with tf.variable_scope('True_Negative'):
        predicted = tf.round(y_pred)
        actual = y_true
        TN = tf.count_nonzero((predicted - 1) * (actual - 1))
    return tf.cast(TN, tf.float32)

def _False_Positive(y_true, y_pred):
    with tf.variable_scope('False_Positive'):
        predicted = tf.round(y_pred)
        actual = y_true
        FP = tf.count_nonzero(predicted * (actual - 1))
    return tf.cast(FP, tf.float32)

def _False_Negative(y_true, y_pred):
    with tf.variable_scope('False_Negative'):
        predicted = tf.round(y_pred)
        actual = y_true
        FN = tf.count_nonzero((predicted - 1) * actual)
    return tf.cast(FN, tf.float32)

def tf_precision(y_true, y_pred):
    with tf.variable_scope('Precision'):
        TP = _True_Positive(y_true, y_pred)
        FP = _False_Positive(y_true, y_pred)
        precision = tf.divide(TP, (TP + FP + epsilon()))
    return precision

def tf_recall(y_true, y_pred):
    with tf.variable_scope('Recall'):
        TP = _True_Positive(y_true, y_pred)
        FN = _False_Negative(y_true, y_pred)
        recall = tf.divide(TP, (TP + FN + epsilon()))
    return recall

def tf_f1(y_true, y_pred):
    with tf.variable_scope('F1_Score'):
        precision = tf.cast(tf_precision(y_true, y_pred), tf.float32)
        recall = tf.cast(tf_recall(y_true, y_pred), tf.float32)
        f1_score = 2 * tf.divide(precision*recall, precision + recall + epsilon())
    return f1_score

def tf_accuracy(y_true, y_pred):
    with tf.variable_scope('Accuracy'):
        TP = _True_Positive(y_true, y_pred)
        TN = _True_Negative(y_true, y_pred)
        FP = _False_Positive(y_true, y_pred)
        FN = _False_Negative(y_true, y_pred)
        accuracy = tf.divide((TP + TN), (TP + TN + FP + FN + epsilon()))
    return accuracy

def tf_specificity(y_true, y_pred):
    with tf.variable_scope('specificity'):
        TN = _True_Negative(y_true, y_pred)
        FP = _False_Positive(y_true, y_pred)
        specificity = tf.divide(TN, (TN + FP))
    return specificity

if __name__ == '__main__':
    y = tf.constant([1., 1., 1., 1., 0., 0.])
    y_ = tf.constant([0.3, 0.0, 0.7, 1, 0.1, 0.9])

    precision = tf_precision(y, y_)
    recall = tf_recall(y, y_)
    f1_score = tf_f1(y, y_)
    accuracy = tf_accuracy(y, y_)
    specificity = tf_specificity(y, y_)
    TP = _True_Positive(y, y_)
    TN = _True_Negative(y, y_)
    FP = _False_Positive(y, y_)
    FN = _False_Negative(y, y_)
    sess = tf.Session()
    print(sess.run([precision, recall, f1_score, accuracy, specificity, TP, TN, FP, FN]))
    sess.close()

