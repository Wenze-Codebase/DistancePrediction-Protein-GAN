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
from matplotlib.pyplot import *



max_list=np.loadtxt('max_130_channel')
min_list=np.loadtxt('min_130_channel')

def preprocess_data_f2d(data):
    with tf.name_scope("preprocess_data_f2d"):
        for pro in range(len(data)):
            for i in range(4):
                data[pro][:,:,i]=data[pro][:,:,i]*(2./(max_list[i]-min_list[i]))+(1-2.*max_list[i]/(max_list[i]-min_list[i]))        
        return data
def preprocess_data_f1d(data):
    with tf.name_scope("preprocess_data_f1d"):
        for pro in range(len(data)):
            for i in range(4,4+62):
                data[pro][:,i-4]=data[pro][:,i-4]*(2./(max_list[i]-min_list[i]))+(1-2.*max_list[i]/(max_list[i]-min_list[i]))        
        return data
def preprocess_data_f0d(data):
    with tf.name_scope("preprocess_data_f0d"):
        for pro in range(len(data)):
            for i in range(128,130):
                data[pro][i-128]=data[pro][i-128]*(2./(max_list[i]-min_list[i]))+(1-2.*max_list[i]/(max_list[i]-min_list[i]))        
        return data

def preprocess_data(data):
    with tf.name_scope("preprocess_data"):
        for pro in range(len(data)):
            for i in range(130):
                data[pro][:,:,:,i]=data[pro][:,:,:,i]*(2./(max_list[i]-min_list[i]))+(1-2.*max_list[i]/(max_list[i]-min_list[i]))        
        return data

def preprocess_lable(lable):
    with tf.name_scope("preprocess_lable"):
        for pro in range(len(lable)):
            lable[pro][lable[pro]>31]=31
            lable[pro]=np.tanh(lable[pro] * (5/12) - (50/12))
        return lable

def deprocess(result):
    with tf.name_scope("deprocess"):
        return (tf.atanh(result) + (50/12)) / (5/12)

def save_images(fetches, epoch, num,a):
    image_dir = os.path.join(a.output_dir, "images")
    print('\n\n\n\nin sava image function \n\n\n\n\n')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    fileset = {"epoch":epoch,"num":num}
    for kind in ["outputs", "targets"]:
        filename= str(num) + "-" +str(epoch)+"-"+ kind 
        filename_png= filename + ".png"
        fileset[kind] = filename_png
        out_path = os.path.join(image_dir, filename_png)
        contents = fetches[kind]
        imshow(contents[0,:,:,0],vmin=4,vmax=16,cmap='gray')
        colorbar()
        savefig(out_path)
        close()
        np.savetxt(os.path.join(image_dir, filename+".np"),contents[0,:,:,0])
    return fileset

def append_index(fileset, a,step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>epoch</th>")
        index.write("<th>index</th><th>output</th><th>target</th></tr>")

    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["epoch"])
    index.write("<td>%s</td>" % fileset["num"])

    for kind in ["outputs", "targets"]:
        index.write("<td><img src='images/%s'></td>" % fileset[kind])

    index.write("</tr>")
    return index_path

