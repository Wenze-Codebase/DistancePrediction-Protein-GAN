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
import layer
from constant import *

def swish(_x, scope=None):
    """swish with paras"""
    with tf.variable_scope(name_or_scope=scope, default_name="swish_with_para"):
        _beta = tf.get_variable("beta", shape=[1],
                                 dtype="float", initializer=tf.constant_initializer(1.))
        return _x * tf.sigmoid(_beta * _x)

def create_generator(generator_inputs, generator_outputs_channels,a):
    l=[]
    generator_inputs,channel_attention, spatial_attention=layer.convolutional_block_attention_module(generator_inputs,0)
    l.append(layer.batchnorm(tf.layers.conv2d(generator_inputs, filters=filters*2,
                                        kernel_size=(3,3),
                                        padding='same',
                                        dilation_rate=(1,1),
                                        activation=swish,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=.1),
                                        bias_initializer=tf.truncated_normal_initializer(stddev=.1),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                        name='Hidden0')))
    for i in range(18):
        l.append(layer.batchnorm(tf.layers.conv2d(l[-1], filters=filters*2,
                                    kernel_size=(7,7),
                                    padding='same',
                                    dilation_rate=(1,1),
                                    activation=swish,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=.1),
                                    bias_initializer=tf.truncated_normal_initializer(stddev=.1),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='Hidden{}'.format(i*3+1))))
      
        l.append(layer.batchnorm(tf.layers.conv2d(l[-1], filters=filters,
                                    kernel_size=(7,7),
                                    padding='same',
                                    dilation_rate=(1,1),
                                    activation=swish,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=.1),
                                    bias_initializer=tf.truncated_normal_initializer(stddev=.1),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='Hidden{}'.format(i*3+2))))
        
        l.append(tf.add(layer.batchnorm(
                    tf.layers.conv2d(l[-1], filters=filters*2,
                                    kernel_size=(7,7),
                                    padding='same',
                                    dilation_rate=(1,1),
                                    activation=swish,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=.1),
                                    bias_initializer=tf.truncated_normal_initializer(stddev=.1),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='Hidden{}-pre'.format(i*3+3))),
                    l[-3],name='Hidden{}'.format(i*3+3)))
    pred=tf.layers.conv2d(l[-1], filters=generator_outputs_channels,
                            kernel_size=(1,1),
                            padding='same',
                            dilation_rate=(1,1),
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=.1),
                            bias_initializer=tf.truncated_normal_initializer(stddev=.1),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                            name='Output')
    realpred=tf.tanh(pred)
    # realpred=(realpred+tf.transpose(realpred,[0,2,1,3]))/2
    # this can be triple for channels, notice that.
    return realpred, channel_attention, spatial_attention

def create_model(inputs, targets, a,learning_rate):
    def create_discriminator(discrim_inputs, discrim_targets, a):
        n_layers = 3
        layers = []
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)
        with tf.variable_scope("layer_1"):
            convolved = layer.discrim_conv(input, a.ndf, stride=2)
            rectified = layer.lrelu(convolved, 0.2)
            layers.append(rectified)
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2 
                convolved = layer.discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = layer.batchnorm(convolved)
                rectified = layer.lrelu(normalized, 0.2)
                layers.append(rectified)
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = layer.discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)
        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs , channel_attention, spatial_attention= create_generator(inputs, out_channels, a)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(inputs, targets, a)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs, outputs, a)

    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(learning_rate, a.beta1)
        discrim_grads_and_vars_1= discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train_1 = discrim_optim.apply_gradients(discrim_grads_and_vars_1)
        with tf.control_dependencies([discrim_train_1]):
            discrim_grads_and_vars_2 = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train_2 = discrim_optim.apply_gradients(discrim_grads_and_vars_2)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train_2]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(learning_rate, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        outputs=outputs,
        channel_attention= channel_attention,
        spatial_attention= spatial_attention,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )
