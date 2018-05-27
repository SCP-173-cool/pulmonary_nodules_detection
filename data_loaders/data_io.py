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
    image = tf.random_crop(image, (36,36,20,1))
    return image, label



def parse_function(example_proto):
    dics = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'image_shape': tf.FixedLenFeature(shape=(4, ), dtype=tf.int64),
        'label': tf.FixedLenFeature([], dtype=tf.int64),
    }
    parsed_example = tf.parse_single_example(example_proto, features = dics)

    image = tf.reshape(tf.decode_raw(
        parsed_example['image'], tf.uint8), parsed_example['image_shape'])
    label = parsed_example['label']

    return image, label


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
            image, label = sess.run([next_element])[0]
            print(label, image.shape)
        except tf.errors.OutOfRangeError:
            break
    writer = tf.summary.FileWriter('test', sess.graph)
    sess.close()
    writer.close()

