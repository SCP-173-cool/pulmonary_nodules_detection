#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables

def _assert(cond, ex_type, msg):
    """A polymorphic assert, works with tensors and boolean expressions.
    If `cond` is not a tensor, behave like an ordinary assert statement, except
    that a empty list is returned. If `cond` is a tensor, return a list
    containing a single TensorFlow assert op.
    Args:
        cond: Something evaluates to a boolean value. May be a tensor.
        ex_type: The exception class to use.
        msg: The error message.
    Returns:
        A list, containing at most one assert op.
    """
    if _is_tensor(cond):
        return [control_flow_ops.Assert(cond, [msg])]
    else:
        if not cond:
            raise ex_type(msg)
        else:
            return []

def _is_tensor(x):
    """Returns `True` if `x` is a symbolic tensor-like object.
    Args:
        x: A python object to check.
    Returns:
        `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
    """
    return isinstance(x, (ops.Tensor, variables.Variable))

def fix_image_flip_shape(image, result):
    """Set the shape to 3 dimensional if we don't know anything else.
    Args:
        image: original image size
        result: flipped or transformed image
    Returns:
        An image whose shape is at least None,None,None.
    """

    image_shape = image.get_shape()
    if image_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None])
    else:
        result.set_shape(image_shape)
    return result

def random_flip_up_down_3D(image, seed=None):
    """Randomly flips an image vertically (upside down).
    With a 1 in 2 chance, outputs the contents of `image` flipped along the first
    dimension, which is `height`.  Otherwise output the image as-is.
    Args:
        image: A 3-D tensor of shape `[height, width, channels].`
        seed: A Python integer. Used to create a random seed. See
        @{tf.set_random_seed}
        for behavior.
    Returns:
        A 3-D tensor of the same type and shape as `image`.
    Raises:
        ValueError: if the shape of `image` not supported.
    """
    with ops.name_scope(None, 'random_flip_up_down_3D', [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = math_ops.less(uniform_random, .5)
        result = control_flow_ops.cond(
            mirror_cond,
            lambda: array_ops.reverse(image, [0]),
            lambda: image,
            name=scope)
        return fix_image_flip_shape(image, result)

def random_flip_left_right_3D(image, seed=None):
    """Randomly flip an image horizontally (left to right).
    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    second dimension, which is `width`.  Otherwise output the image as-is.
    Args:
        image: A 3-D tensor of shape `[height, width, channels].`
        seed: A Python integer. Used to create a random seed. See
        @{tf.set_random_seed}
        for behavior.
    Returns:
        A 3-D tensor of the same type and shape as `image`.
    Raises:
        ValueError: if the shape of `image` not supported.
    """
    with ops.name_scope(None, 'random_flip_left_right_3D', [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = math_ops.less(uniform_random, .5)
        result = control_flow_ops.cond(
            mirror_cond,
            lambda: array_ops.reverse(image, [1]),
            lambda: image,
            name=scope)
        return fix_image_flip_shape(image, result)

def random_flip_front_end_3D(image, seed=None):
    """Randomly flip an image horizontally (front to end).
    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    second dimension, which is `width`.  Otherwise output the image as-is.
    Args:
        image: A 3-D tensor of shape `[height, width, channels].`
        seed: A Python integer. Used to create a random seed. See
        @{tf.set_random_seed}
        for behavior.
    Returns:
        A 3-D tensor of the same type and shape as `image`.
    Raises:
        ValueError: if the shape of `image` not supported.
    """
    with ops.name_scope(None, 'random_flip_front_end_3D', [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = math_ops.less(uniform_random, .5)
        result = control_flow_ops.cond(
            mirror_cond,
            lambda: array_ops.reverse(image, [2]),
            lambda: image,
            name=scope)
        return fix_image_flip_shape(image, result)

def random_rotate(image, name=None, seed=None):
    """Rotate image(s) counter-clockwise by 90 degrees.
    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or
            3-D Tensor of shape `[height, width, channels]`.
        k: A scalar integer. The number of times the image is rotated by 90 degrees.
        name: A name for this operation (optional).
    Returns:
        A rotated tensor of the same type and shape as `image`.
    Raises:
        ValueError: if the shape of `image` not supported.
    """ 
    with ops.name_scope(name, 'random_rotate', [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = math_ops.less(uniform_random, .5)
        result = control_flow_ops.cond(
            mirror_cond,
            lambda: array_ops.transpose(array_ops.reverse_v2(image, [1]), [1, 0 ,2, 3]),
            lambda: image,
            name=scope)
        return fix_image_flip_shape(image, result)

def central_crop(image, crop_size):
    """Crop the central region of the image.
    Remove the outer parts of an image but retain the central region of the image
    along each dimension. If we specify central_fraction = 0.5, this function
    returns the region marked with "X" in the below diagram.
        --------
        |        |
        |  XXXX  |
        |  XXXX  |
        |        |   where "X" is the central 50% of the image.
        --------
    Args:
        image: 3-D float Tensor of shape [height, width, depth]
        central_fraction: float (0, 1], fraction of size to crop
    Raises:
        ValueError: if central_crop_fraction is not within (0, 1].
    Returns:
        3-D float Tensor
    """
    crop_size_f = [float(i) for i in crop_size]
    with ops.name_scope(None, 'central_crop', [image]):
        image = ops.convert_to_tensor(image, name='image')
        

        img_shape = array_ops.shape(image)
        depth = image.get_shape()[3]
        img_h = math_ops.to_double(img_shape[0])
        img_w = math_ops.to_double(img_shape[1])
        img_d = math_ops.to_double(img_shape[2])
        bbox_h_start = math_ops.to_int32((img_h - crop_size_f[0]) / 2)
        bbox_w_start = math_ops.to_int32((img_w - crop_size_f[1]) / 2)
        bbox_d_start = math_ops.to_int32((img_d - crop_size_f[2]) / 2)

        bbox_start = [bbox_h_start, bbox_w_start, bbox_d_start]

        bbox_begin = array_ops.stack(bbox_start + [0])
        bbox_size = array_ops.stack(crop_size + [-1])
        image = array_ops.slice(image, bbox_begin, bbox_size)

        # The first two dimensions are dynamic and unknown.
        image.set_shape([None, None, None, depth])
        return image
