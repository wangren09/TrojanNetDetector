#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf
import math
import dataset_input
import resnet
import sys
import os
import pickle
from collections import Counter
import utilities
import json

BINARY_SEARCH_STEPS = 8  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 101  # number of iterations to perform gradient descent
#ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
INITIAL_CONST = 0.01     # the initial constant lambda to pick as a first guess

class PerImgPert:
        def __init__(self, sess, config, filepath, batch_size, regu,
                 learning_rate = 0.1,
                 binary_search_steps = 1, max_iterations = 101,
                 initial_const = 1):
            
            

            


            
            
            
            
            image_size, num_channels, num_labels = 32, 3, 10
            self.sess = sess
            self.LEARNING_RATE = learning_rate
            self.MAX_ITERATIONS = max_iterations
            self.BINARY_SEARCH_STEPS = binary_search_steps
            #self.ABORT_EARLY = abort_early
            self.initial_const = initial_const
            self.batch_size = batch_size
            
            shape = (batch_size,image_size,image_size,num_channels)
            shape_pert = (batch_size,image_size,image_size,num_channels)

            
            # these are variables to be more efficient in sending data to tf
            self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
            self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
            self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
            self.modifier = tf.Variable(np.zeros(shape_pert,dtype=np.float32))
            self.det = tf.Variable(np.ones(shape_pert,dtype=np.float32), constraint = lambda x:tf.clip_by_value(x, 0, 255))

            
            self.newimg = self.modifier*self.det + self.timg*(1-self.modifier)
            self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels), name="tlab")
            self.assign_timg = tf.placeholder(tf.float32, shape, name="timag")
            self.assign_const = tf.placeholder(tf.float32, [batch_size], name="tconst")
            # the variable we're going to optimize over
            
            
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            #with tf.variable_scope(reuse=tf.AUTO_REUSE):
            model = resnet.Model(config.model, self.newimg)
                   

            # Setting up the Tensorboard and checkpoint outputs
            model_dir = filepath

            #saver = tf.train.Saver()
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            var_list = var_list[5:]
            
            
            #saver = tf.train.Saver(var_list=[v for v in all_variables if not in v.name])
            saver = tf.train.Saver(max_to_keep=3, var_list=var_list)     
            
            
            latest_checkpoint = tf.train.latest_checkpoint(model_dir)

            if latest_checkpoint is not None:
                saver.restore(sess, latest_checkpoint)
                print('Restoring last saved checkpoint: ', latest_checkpoint)
            else:
                print('Check model directory')
                exit()

            

            self.output=model.pre_softmax
 

            # compute the probability of the label class versus the maximum other
            real = tf.reduce_sum((self.tlab)*self.output,1)

            other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)

            loss1 = tf.maximum(-15.0, other -  real)
            self.loss1 = tf.reduce_sum(loss1)
            
            if regu == "l2":
                self.loss2 = tf.reduce_sum(tf.square(self.modifier))
            else:
                self.loss2 = tf.reduce_sum(tf.abs(self.modifier))

            self.loss = self.loss1
            self.modifier_ST = tf.clip_by_value(tf.sign(self.modifier)*tf.maximum(tf.abs(self.modifier) - self.LEARNING_RATE/self.initial_const, 0), clip_value_min=0, clip_value_max=1)
            self.assign_ST = tf.assign(self.modifier, self.modifier_ST)
        
            # Setup the adam optimizer and keep track of variables we're creating
            start_vars = set(x.name for x in tf.global_variables())
            optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
            self.train1= optimizer.minimize(self.loss, var_list=[self.modifier])
            self.train2 = optimizer.minimize(self.loss, var_list=[self.det])
            end_vars = tf.global_variables()
            new_vars = [x for x in end_vars if x.name not in start_vars]
            
            
            # these are the variables to initialize when we run
            self.setup = []
            self.setup.append(self.timg.assign(self.assign_timg))
            self.setup.append(self.tlab.assign(self.assign_tlab))
            self.setup.append(self.const.assign(self.assign_const))

            #self.init = tf.variables_initializer(var_list=[self.modifier]+new_vars)
            self.init = tf.variables_initializer(var_list=[self.modifier]+[self.det]+new_vars)
            
        def attack(self, imgs, labs):
            self.assign_input = imgs
            batch_size = self.batch_size
            best_loss = self.loss
            perturb_best = self.modifier
            

            CONST = np.ones(batch_size)*self.initial_const

            for outer_step in range(self.BINARY_SEARCH_STEPS):
                self.sess.run(self.init)

                self.sess.run(self.setup, {self.assign_timg: imgs,
                                       self.assign_tlab: labs,
                                       self.assign_const: CONST})
                   
                for iteration in range(self.MAX_ITERATIONS):
                    
                    # perform the attack 
                    _, outp = self.sess.run([self.train1, self.output])
                    self.sess.run(self.assign_ST)
                    _, perturb_temp, det, outp = self.sess.run([self.train2, self.modifier, self.det, self.output])

                    if iteration%(self.MAX_ITERATIONS//10) == 0:
                        print(iteration,self.sess.run((self.loss,self.loss1,self.loss2,tf.reduce_sum(tf.abs(perturb_temp)))))
                        #print(iteration,self.sess.run((self.loss,self.loss1,self.loss2,tf.reduce_sum(tf.abs(perturb_temp)),tf.reduce_sum(tf.abs(perturb_temp*det)))))
                        #print(self.timg[0].eval())
                    if iteration%50 == 0:#if iteration == 200 or iteration == 50:
                        print(np.argmax(outp, axis = 1))

            output_arg1 = np.argmax(outp, axis = 1)
            indices1 = np.where(np.equal(output_arg1, np.argmax(labs, axis = 1)))
            return perturb_temp, det, indices1, outp

