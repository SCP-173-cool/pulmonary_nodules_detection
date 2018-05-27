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


def augmentation(image, label):
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

    return image, label


def data_loader(tfrecord_lst,
                num_repeat=1,
                shuffle=False,
                batch_size=128,
                num_processors=4,
                mode='train',
                name=""):

    with tf.VariableScope(reuse=False, name=name):
        dataset = tf.data.TFRecordDataset(filename_lst)
        new_dataset = dataset.map(
            parse_function, num_parallel_calls=num_processors)
        if shuffle:
            new_dataset = new_dataset.shuffle(buffer_size=100000)
        new_dataset = new_dataset.repeat(epoch)


    return


if __name__ == '__main__':
    filename_lst = ['./output/train.tfrecord']
    dataset = tf.data.TFRecordDataset(filename_lst)
    new_dataset = dataset.map(parse_function, num_parallel_calls=4)
    new_dataset = new_dataset.shuffle(buffer_size=100000).repeat(8)
    new_dataset = new_dataset.map(augmentation, num_parallel_calls=4)
    new_dataset = new_dataset.batch(4)

    iterator = new_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess = tf.InteractiveSession()
    for i in range(22):
        try:
            image = sess.run([next_element])
            print(image)
        except tf.errors.OutOfRangeError:
            break
    writer = tf.summary.FileWriter('test', sess.graph)
    sess.close()
    writer.close()
