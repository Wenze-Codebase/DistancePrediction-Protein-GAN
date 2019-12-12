from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
from constant import *


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def combined_static_and_dynamic_shape(tensor):
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def convolutional_block_attention_module(feature_map, index, inner_units_ratio=0.5):
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        channel_avg_weights = tf.reduce_mean(tf.reduce_mean(feature_map,axis=2,keepdims=True), axis=1, keepdims=True)
        channel_max_weights = tf.reduce_max(tf.reduce_max(feature_map,axis=2,keepdims=True), axis=1, keepdims=True)
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)
        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=None
        )
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]],name='channel_attention')
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)
        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        spatial_attention = tf.layers.conv2d(channel_wise_pooling, filters=1,
                                    kernel_size=(7,7),
                                    padding='same',
                                    dilation_rate=(1,1),
                                    activation=tf.nn.sigmoid,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=.1),
                                    bias_initializer=tf.truncated_normal_initializer(stddev=.1),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='spatial_attention')
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
        return feature_map_with_attention, channel_attention[0,0,0,:], spatial_attention[0,:,:,0]