#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'data_loaders')
sys.path.insert(0, 'models')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from models.model_main import MODEL
from data_loaders.data_io import data_loader


import tensorflow as tf

train_tfrecord_lst = ['./output/LUNA/train.tfrecord']
valid_tfrecord_lst = ['./output/LUNA/valid.tfrecord']
test_tfrecord_lst = ['./output/LUNA/test.tfrecord']

train_loader = data_loader(train_tfrecord_lst,
                           num_repeat=12,
                           shuffle=True,
                           batch_size=2,
                           num_processors=4,
                           augmentation=True,
                           name='train_dataloader')

valid_loader = data_loader(valid_tfrecord_lst,
                           num_repeat=1,
                           shuffle=False,
                           batch_size=1,
                           num_processors=4,
                           augmentation=True,
                           name='valid_dataloader')

test_loader = data_loader(test_tfrecord_lst,
                           num_repeat=1,
                           shuffle=False,
                           batch_size=1,
                           num_processors=4,
                           augmentation=False,
                           name='test_dataloader')

model = MODEL([36, 36, 20], 1)
model.build_model()



sess = tf.Session()
saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

tvars = tf.trainable_variables()
print("Trainable Parameters:")
for tvar in tvars:
    print(tvar.name)
print("=================================================================")

learning_rate_placeholder = tf.placeholder(tf.float32, shape=[], name='learning_rate')
grads = tf.gradients(model.loss, tvars)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_placeholder, momentum=0.9)
train_op = optimizer.apply_gradients(zip(grads, tvars))

sess.run(tf.global_variables_initializer())

global_iteration_count = 0

for i in range(12):

    image_batch, label_batch = sess.run([train_loader])[0]
    train_feed = {model.image: image_batch, 
                  model.label: label_batch,
                  learning_rate_placeholder: 0.001}
    _, loss = sess.run([train_op, model.loss], feed_dict=train_feed)
    print(loss)

            



