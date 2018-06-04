#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from image3D_ops import *


def augmentation_function(image, label):
    image = random_flip_up_down_3D(image)
    image = random_flip_left_right_3D(image)
    image = random_flip_front_end_3D(image)
    image = random_rotate(image, name='random_rot')

    image = tf.random_crop(image, (36, 36, 20, 1))
    image = tf.divide(image, tf.constant(255.0, dtype=tf.float32))
    image = tf.image.random_contrast(image, 0, 1)
    image = tf.image.random_brightness(image, 0.3)

    return image, label


def parse_function(example_proto):
    dics = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'image_shape': tf.FixedLenFeature(shape=(4, ), dtype=tf.int64),
        'label': tf.FixedLenFeature([], dtype=tf.int64),
    }
    parsed_example = tf.parse_single_example(example_proto, features=dics)

    image = tf.reshape(tf.decode_raw(
        parsed_example['image'], tf.uint8), parsed_example['image_shape'])
    label = parsed_example['label']

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    return image, label


def data_loader(tfrecord_lst,
                num_repeat=1,
                shuffle=False,
                batch_size=128,
                num_processors=4,
                augmentation=False,
                name="",
                device='cpu:0'):

    with tf.variable_scope(name, reuse=False):
        with tf.device('/{}'.format(device)):
            dataset = tf.data.TFRecordDataset(tfrecord_lst)
            new_dataset = dataset.map(
                parse_function, num_parallel_calls=num_processors)

            if shuffle:
                new_dataset = new_dataset.shuffle(buffer_size=100000)
            if augmentation:
                new_dataset = new_dataset.map(augmentation_function,
                                    num_parallel_calls=num_processors)
            new_dataset = new_dataset.repeat(num_repeat)
            new_dataset = new_dataset.batch(batch_size)

            iterator = new_dataset.make_initializable_iterator()
            #next_element = iterator.get_next()

    #return 
    return iterator


if __name__ == '__main__':
    train_tfrecord_lst = ['../output/LUNA/train.tfrecord']
    test_tfrecord_lst = ['../output/LUNA/test.tfrecord']
    train_loader = data_loader(train_tfrecord_lst,
                               num_repeat=1,
                               shuffle=True,
                               batch_size=1024,
                               num_processors=4,
                               augmentation=True,
                               name='train_dataloader')


    test_loader = data_loader(test_tfrecord_lst,
                               num_repeat=1,
                               shuffle=False,
                               batch_size=1,
                               num_processors=4,
                               augmentation=False,
                               name='test_dataloader')


    sess = tf.InteractiveSession()
    sess.run(train_loader.initializer)
    for i in range(2200):
        try:
            image, label = sess.run([train_loader.get_next()])[0]
            print(sum(label), image.shape)
        except tf.errors.OutOfRangeError:
            print 'OOR'
            sess.run(train_loader.initializer)
    writer = tf.summary.FileWriter('test', sess.graph)
    sess.close()
    writer.close()
