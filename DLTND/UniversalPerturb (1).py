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


class UniversalPert:
        def __init__(self, sess, config, filepath, batch_size, batch_size_move, regu,
                 learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 initial_const = INITIAL_CONST):
            


            
            
            
            
            image_size, num_channels, num_labels = 32, 3, 10
            self.sess = sess
            self.LEARNING_RATE = learning_rate
            self.MAX_ITERATIONS = max_iterations
            self.BINARY_SEARCH_STEPS = binary_search_steps
            #self.ABORT_EARLY = abort_early
            self.initial_const = initial_const
            self.batch_size = batch_size
            self.batch_size2 = batch_size_move
            batch_size_full = 50
            
            shape = (batch_size_full,image_size,image_size,num_channels)
            shape_pert = (1,image_size,image_size,num_channels)

            
            # these are variables to be more efficient in sending data to tf
            self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
            self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
            self.tlab2 = tf.Variable(np.zeros((batch_size_move,num_labels)), dtype=tf.float32)
            self.const = tf.Variable(np.zeros(1), dtype=tf.float32)
            self.const2 = tf.Variable(np.zeros(batch_size_move), dtype=tf.float32)
            self.modifier = tf.Variable(np.zeros(shape_pert,dtype=np.float32))
            self.det = tf.Variable(np.ones(shape_pert,dtype=np.float32), constraint = lambda x:tf.clip_by_value(x, 0, 255))


            
            self.newimg = self.modifier*self.det + self.timg*(1-self.modifier)

            
            self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels), name="tlab")
            self.assign_tlab2 = tf.placeholder(tf.float32, (batch_size_move,num_labels), name="tlab2")
            self.assign_timg = tf.placeholder(tf.float32, shape, name="timag")
            self.assign_const = tf.placeholder(tf.float32, [1], name="tconst")
            self.assign_const2 = tf.placeholder(tf.float32, [batch_size_move], name="tconst2")

            
            
            
            global_step = tf.contrib.framework.get_or_create_global_step()

            model = resnet.Model(config.model, self.newimg)
                   

            # Setting up the Tensorboard and checkpoint outputs
            model_dir = filepath

            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            var_list = var_list[7:]
                        
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
            real = tf.reduce_sum((self.tlab)*self.output[0:batch_size],1)

            other = tf.reduce_max((1-self.tlab)*self.output[0:batch_size] - (self.tlab*10000),1)
            
            real2 = tf.reduce_sum((self.tlab2)*self.output[batch_size:batch_size_full],1)

            other2 = tf.reduce_max((1-self.tlab2)*self.output[batch_size:batch_size_full] - (self.tlab2*10000),1)

            self.loss1d = tf.maximum(-15.0, real - other)
            self.loss1 = tf.reduce_sum(self.loss1d)
            
            self.loss11d = tf.maximum(-15.0, other2 - real2)
            self.loss11 = tf.reduce_sum(self.loss11d)
            
            if regu == "l2":
                self.loss2 = tf.reduce_sum(tf.square(self.modifier))
            else:
                self.loss2 = tf.reduce_sum(tf.abs(self.modifier))

            self.loss = self.loss1 + self.loss11
            self.modifier_ST = tf.clip_by_value(tf.sign(self.modifier)*tf.maximum(tf.abs(self.modifier) - self.LEARNING_RATE/self.const,0),                                                clip_value_min=0, clip_value_max=1)
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
            self.setup.append(self.tlab2.assign(self.assign_tlab2))
            self.setup.append(self.const2.assign(self.assign_const2))
            self.init = tf.variables_initializer(var_list=[self.modifier]+[self.det]+new_vars)
            
        def attack(self, imgs, labs, labs2):
            self.assign_input = imgs
            batch_size = self.batch_size
            batch_size2 = self.batch_size2
            batch_size_full = batch_size + batch_size2
            best_loss = self.loss
            perturb_best = self.modifier
            upper = 6
            lower = 0
            area_best = 1e8
            l1_area = []
            logits = []
            
            index_const = -1
            
            self.sess.run(self.init)

            for outer_step in range(self.BINARY_SEARCH_STEPS):

                CONST = np.ones(1)*self.initial_const
                CONST2 = np.ones(batch_size2)*self.initial_const

                if index_const == -1 and outer_step > 0:
                    self.sess.run(self.init)
                    lower = self.initial_const
                    self.initial_const = (self.initial_const + upper)/2
                    CONST = np.ones(1)*self.initial_const
                    CONST2 = np.ones(batch_size2)*self.initial_const
                if index_const == 1:
                    self.sess.run(self.init)
                    upper = self.initial_const
                    self.initial_const = (self.initial_const + lower)/2
                    index_const = -1
                    CONST = np.ones(1)*self.initial_const
                    #print(index_const)
                
                self.sess.run(self.setup, {self.assign_timg: imgs,
                                       self.assign_tlab: labs,
                                       self.assign_const: CONST,
                                       self.assign_tlab2: labs2,
                                       self.assign_const2: CONST2})
    

    
                
                for iteration in range(self.MAX_ITERATIONS):
                    
                    # perform the attack 
                    _, outp = self.sess.run([self.train1, self.output])

                    self.sess.run(self.assign_ST)

                    
                    _, perturb_temp, det, loss_comp1, loss_comp2, outp = self.sess.run([self.train2, self.modifier, self.det, self.loss1d, self.loss11d, self.output])

                    num_val = sum([1 for val in loss_comp1 if val <= 0]) + sum([1 for val in loss_comp2 if val <= 0])
                    if num_val > batch_size_full*0.7:
                        index_const = 1
                    if iteration%(self.MAX_ITERATIONS//10) == 0:
                        print(iteration,self.sess.run((self.loss,self.loss1,self.loss2,tf.reduce_sum(tf.abs(perturb_temp)))))
                        #print(iteration,self.sess.run((self.loss,self.loss1,self.loss2,tf.reduce_sum(tf.abs(perturb_temp)),tf.reduce_sum(tf.abs(perturb_temp*det)))))
                    if iteration%50 == 0:#iteration == 100 or iteration == 50:
                        print(np.argmax(outp, axis = 1))
                print(index_const)
                print(self.initial_const)
                
                if index_const == 1 or outer_step == 0:
                    area_temp = np.sum(np.abs(perturb_temp))
                    logits.append(outp)
                    l1_area.append(area_temp)
                    if area_best > area_temp:
                        best_out = outp[:]
                        #print(np.argmax(best_out, axis = 1))
                        print(area_temp)
                        if area_temp > 0.5:
                            area_best = area_temp
                            print(area_temp)
            output_arg1 = np.argmax(outp[0:batch_size], axis = 1)
            output_arg2 = np.argmax(outp[batch_size:], axis = 1)

            indices1 = np.where(np.equal(output_arg1, np.argmax(labs, axis = 1)))
            indices2 = np.where(np.not_equal(output_arg2, np.argmax(labs2, axis = 1)))
            return area_best, det, indices1, indices2, best_out, logits, l1_area

