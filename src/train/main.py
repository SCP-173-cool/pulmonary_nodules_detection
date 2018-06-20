#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktarxiao
"""

from __future__ import print_function
import sys
sys.dont_write_bytecode = True

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import tensorflow as tf
from trainer import base_trainer





if __name__ == "__main__":
    trainer = base_trainer()
    sess = tf.Session()
    trainer.train(sess)
    sess.close()
