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
import function
from constant import *

def getdata(file_path):
    dataf0=[]
    dataf1=[]
    dataf2=[]
    labels=[]
    for serialized_example in tf.python_io.tf_record_iterator(file_path):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        index = example.features.feature['index'].int64_list.value[0]

        name = example.features.feature['name'].bytes_list.value[0]        

        length = example.features.feature['length'].int64_list.value[0]

        count = example.features.feature['count'].int64_list.value[0]

        ccmpred = np.fromstring(example.features.feature['ccmpred'].bytes_list.value[0], dtype=np.float32)
        ccmpred.resize((length,length))
        mi = np.fromstring(example.features.feature['mi'].bytes_list.value[0], dtype=np.float32)
        mi.resize((length,length))

        deepcnf = np.fromstring(example.features.feature['deepcnf'].bytes_list.value[0], dtype=np.float32)
        deepcnf.resize((length,3+8))

        seql = np.fromstring(example.features.feature['seq'].bytes_list.value[0], dtype=np.float32)
        seql.resize((length,20))


        freq = np.fromstring(example.features.feature['freq'].bytes_list.value[0], dtype=np.float32)
        freq.resize((length,21))
        gap = np.fromstring(example.features.feature['gap'].bytes_list.value[0], dtype=np.float32)
        gap.resize((length,length))

        pos = np.fromstring(example.features.feature['pos'].bytes_list.value[0], dtype=np.int32)
        pos.resize((length,length))

        spd = np.fromstring(example.features.feature['spd'].bytes_list.value[0], dtype=np.float32)
        spd.resize((length,10)) 

        label = np.fromstring(example.features.feature['label'].bytes_list.value[0], dtype=np.float32)
        label.resize((length,length))

        f0d = np.array([length, count]).astype(np.float32)
        f1d = np.concatenate((deepcnf, seql, freq, spd), axis=1).astype(np.float32)
        f2d = np.stack((ccmpred, mi, gap, pos), axis=2).astype(np.float32)

        dataf0.append(f0d)
        dataf1.append(f1d)
        dataf2.append(f2d)
        labels.append(label)

    return dataf0,dataf1,dataf2,labels

def load_examples(train_file_path):
    dataf0,dataf1,dataf2,labels=getdata(train_file_path)

    dataf0=function.preprocess_data_f0d(dataf0)
    dataf1=function.preprocess_data_f1d(dataf1) 
    dataf2=function.preprocess_data_f2d(dataf2) 
    labels=function.preprocess_lable(labels)
    
    return dataf0,dataf1,dataf2,labels

