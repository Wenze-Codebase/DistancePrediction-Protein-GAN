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
import model, function, data
from constant import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="GAN_train.tfrecords",help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "export"])
parser.add_argument("--output_dir", default="outputs", help="where to put output files")
parser.add_argument("--seed", type=int, default=58, help="seed of random progress")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing, if default None, use arg:output_dir to store models")
parser.add_argument("--max_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--ndf", type=int, default=128, help="number of discriminator filters in first conv layer")
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=258.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
a = parser.parse_args()

def main(_):
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")       
    
    print("Notice the arguments of this run is as following : ")
    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        batch_input = tf.placeholder("float", shape=[1, None, None, input_chanel_num], name='Input_without_preprocess')
        
        with tf.variable_scope("generator"):
            batch_output,_,_ = model.create_generator(batch_input,1,a)
            output = function.deprocess(batch_output)
            real_output=tf.identity(output,name='Final_output')

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    dataf0,dataf1,dataf2,labels = data.load_examples(a.input_dir)

    batch_input = tf.placeholder("float", shape=[1, None, None, input_chanel_num], name='Input')
    batch_target = tf.placeholder("float", shape=[1, None, None, 1], name='Target') 
    learning_rate = tf.placeholder("float", shape=[])

    model_using = model.create_model(batch_input, batch_target,a,learning_rate)

    targets =function.deprocess(batch_target)
    outputs = function.deprocess(model_using.outputs) 

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=100)

    logdir = a.output_dir 
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        start = time.time()
        logfile=open(a.output_dir+'/logfile','a')

        for epoch in range(a.max_epochs):
            if epoch<21:
                lr_using=a.lr
            elif epoch<51:
                lr_using=a.lr * 0.1
            else:
                lr_using=a.lr *0.01

            logfile.write('Epoch '+str(epoch)+': learning_rate: '+str(lr_using)+'\n')
            print('Epoch'+str(epoch))
            discrim_loss_train=[]
            gen_loss_GAN_train=[]
            gen_loss_L1_train=[]
            
            fetches = {
                    "train": model_using.train,
                    "global_step": sv.global_step,
                }
            fetches["discrim_loss"] = model_using.discrim_loss
            fetches["gen_loss_GAN"] = model_using.gen_loss_GAN
            fetches["gen_loss_L1"] = model_using.gen_loss_L1
 
            for index in range(training_protein_nums):
                length=dataf2[index].shape[1]
                if length  <max_length:
                    input_data= np.concatenate([dataf2[index],np.tile(dataf1[index][np.newaxis], [length, 1, 1]),
                        np.tile(dataf1[index][:, np.newaxis], [1, length, 1]),np.tile(dataf0[index][np.newaxis, np.newaxis], [length, length, 1]),
                        ], axis=2)[np.newaxis]
                    label=labels[index][np.newaxis,:,:,np.newaxis]
                    print(epoch,index)
                    if index%1000==0:
                        print(a.output_dir)
		    results = sess.run(fetches,feed_dict = {batch_input:input_data,batch_target:label,learning_rate:lr_using})

		    discrim_loss_train.append( results["discrim_loss"])
		    gen_loss_GAN_train.append( results["gen_loss_GAN"])
		    gen_loss_L1_train.append( results["gen_loss_L1"])
	    logfile.write('  trainging:\n')
	    logfile.write('    discrim_loss:{}\n'.format(sum(discrim_loss_train)/len(discrim_loss_train)))
	    logfile.write('    gen_loss_GAN:{}\n'.format(sum(gen_loss_GAN_train)/len(gen_loss_GAN_train)))
	    logfile.write('    gen_loss_L1:{}\n'.format(sum(gen_loss_L1_train)/len(gen_loss_L1_train)))
	    logfile.flush()

	           
	    print("saving model of epoch"+str(epoch))
	    saver.save(sess, os.path.join(a.output_dir, "model"+str(epoch)))

	    if sv.should_stop():
	        break
        logfile.close()

if __name__ == '__main__':  
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    tf.app.run(main=main)
