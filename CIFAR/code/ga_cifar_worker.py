# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:08:12 2020

@author: 
"""

'''
Copyright 2022 by Anonymous Authors
This code is protected by the Academic Free license Version 3.0
See LICENSE.md in the repository root folder for the terms of this license
'''

'''
Genetic algorithm code for evolving adversarial examples for CIFAR-10 images
  - the ga_control.py program calls this program as a command line program and among the arguments
    it sends is an ID for a CIFAR wmage for which this program will evolve an adversarial example
  - The population images are in numpy arrays of shape (pop_size, 3072) and the CIFAR traget images is of shape (3072,)
    * Cifar images are 32x32 for a total of 1024 pixels, with RGB components for each pixel, implying 3072 wlwmwnts for each image
    * Viewing images requires the 3072 numerical elements be reshaped: teh first 1034 values are red intensities, 
      the next 1024 values are green intensities and the final 1024 values are blue intensities.
'''

#import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from matplotlib.colors import hsv_to_rgb
#from numba import njit
import random

#from keras import models
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.datasets import cifar10
#from keras.utils import to_categorical
#from keras.datasets import mnist

import numpy as np
#import json
import argparse
#import os
import datetime
import re
from sys import exit

def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError 

class GA():
    
    #def __init__(self, pop_size, num_gen, prob_mut_genome, prob_mut_pixel, prob_wht, prob_blk, gen, target_index, input_folder, output_folder, loaded_model, max_img, log_name, fit_type = 'mad', min_mad = 0.1, rand_mode = 'rand'):
    def __init__(self, pop_size, num_gen, prob_mut_genome, prob_mut_pixel, mut_light_bias, gen, folder_base, input_folder, model_filename, 
                 output_folder, max_img, fit_type='mad-linear', select_type='rank-linear', factor_rank_nonlinear=0.9, 
                 min_mad = 0.1, max_mad = 0.15, rand_mode = 'rand', gpu_mode='True', out_only_best=True):
        
        ''' Import matplotlib if graphs to be shwon '''
        #if max_img:
        #    import matplotlib.pyplot as plt
        
        
        ''' Set general parameters '''
        self.pop_size = pop_size
        self.num_gen = num_gen
        self.min_mad = min_mad
        self.fit_type = fit_type
        self.select_type = select_type
        self.rank_nonlin_factor = factor_rank_nonlinear
        self.max_mad = max_mad #0.15
        self.rand_mode = rand_mode
        self.gpu_mode = str_to_bool(gpu_mode)
        #self.log_name = log_name
        #self.rand_mode = rand_mode
        ''' re object for stripping filename characters '''
        self.re_strip = re.compile('_B\d+E\d+')
        self.train_images_num = 50000
        self.num_pixels_rgb = 3072
        self.num_pixels = 1024
        self.num_rgb = 3
        self.gen = 0  # Generation number during evolution
        self.thres_pix_diff = 1e-7  # Threshold for calling a pixel different from a target pixel
        self.write_adv_eg = True
        self.init_pop_mode = 'all'  # or 'trunc'; truncate initial adv eg population to a priori pop_size or keep all
        ''' Set random population parameters'''
        self.prob_mut_genome = prob_mut_genome
        self.prob_mut_pixel = prob_mut_pixel
        self.mut_light_bias = mut_light_bias  # 0 = all mutations toward target pixel, 0.5 = half toward, half away from target pixel, 1.0 = all away
        #self.mut_bias = 0.1  # 0 = all mutations toward target pixel, 0.5 = half toward, half away from target pixel, 1.0 = all away
        # Settings before experiment with introducing light_mut_bias (prob_mut_genome, prob_mut_pixel, self.mut_bias) = (1.0, 0.5, 0.0)
        
        ''' Settings for image/data storage '''
        self.in_folder = input_folder
        self.out_folder = output_folder
        self.write_pop_gen = False        # IF True, write log file for population initialization
        self.max_img = max_img            # If True, then prints image of best-fit and other messaging
        self.img_dump = False   # Flag for writing initial population to numpy file
        #self.img_read = False  # Flag for reading initial population from numpy file
        self.img_dump_file = 'pop_%d.npy'
        self.image_folder = 'images/'
        self.folder_base = folder_base
        ''' Placeholders for, in this order, CIFAR ID, 100*mut_genome, 100*mut_pixel, 100*mut_light_bias, serial # with these settings '''
        #self.best_fit_file = 'best_fit_%d_%d_%d_%d_%d.npy'
        self.best_dump = True
        self.db_out = True
        self.out_only_best = True
        self.scen_name = '_'.join([model_filename.rstrip('.h5'), self.fit_type, self.select_type, self.rand_mode])
        self.best_fit_file = self.scen_name + '_bf_%d.npy'
        
        ''' These variables are not needed for CIFAR data as they were for MNIST '''
        '''
        self.prob_wht = prob_wht
        self.prob_blk = prob_blk
        '''
        
        
        ''' Load Model & Weights, and Compile '''
        '''
        json_file = open(input_folder + model_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = models.model_from_json(loaded_model_json)
        self.loaded_model.load_weights(input_folder + weights_filename)
        self.loaded_model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        '''
        self.loaded_model = models.load_model(input_folder + model_filename) #models.load_model(input_folder + model_filename + '.h5')
        self.loaded_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
        '''
        if self.gpu_mode:
            with tf.device('/gpu:0'):
                self.loaded_model = models.load_model(input_folder + model_filename + '.h5')
                self.loaded_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
        else:
            with tf.device('/cpu:0'):
                self.loaded_model = models.load_model(input_folder + model_filename + '.h5')
                self.loaded_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy']) '''

        if self.max_img:
            print("Model loaded from disk and compiled")
            self.loaded_model.summary()

        
        ''' Load CIFAR data '''
        ''' Old, working code to laod images from the original CIFAR data source 
            Beware: the data loaded from this data source are in a different format then
            when loaded from tensorflow.  Data load in the manner below require the cifar2show()
            function to format the data properly for display and submitting the data in this format
            is not appropriate for the neural network model, which was trained on the data as 
            formatted from tensorflow. 
            
            Also, the data will be of shape (50000, 32, 32, 3) rather than (50000, 3072) 
            
            Each genome is a numpy array of dimension (3072,). The (32, 32, 3) format is 
            needed for displaying an image and for input for prediction with the neural network. 
            The tactics for managing the different data structure requirements is as follows:
                - Keep each iamge in self.train_images as (1024, 3) np.uint8 integers
                  to save memory
                  * 3 channels for facilitate easy generation of population and mutation operation
                    while keeping the combinations of RGB values from the training images intact
                - Convert images to (1024, 3) np.float32 arrays and divide by 255 when put into 
                  the population for the GA
                - Reshape images to (32, 32, 3) np.float32 for prediction
                - For troubleshooting structure, trace these variables:
                    * self.train_images
                    * self.pop
                    * self.target_img
                    * self.check_target_class_pop
                - For computation os MAD, the shape of the training images is changed to (50000, 3072) 
                  for purposes of this computation
                - pop_fit() requires a population shape of (-1,3072)
            '''
        '''
        filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', ]
        self.train_labels = []
        self.train_images = np.array([]).astype(np.uint8).reshape(0,3072)
        for filename in filenames:
            data = self.unpickle(self.in_folder + filename)
            self.train_images = np.concatenate((self.train_images, data[b'data']))
            self.train_labels += np.array(data[b'labels'])
        del data
        assert self.train_images.shape == (self.train_images_num, self.num_pixels_rgb)
        '''
        
        ''' Note that the pixels values loaded in the statement below are np.uint8 integers '''
        ''' load CIFAR from file '''
        ''' Replace loading of resident files with download from tensorflow.keras.datasets 
        self.train_images = np.load(input_folder + 'cifar_train_images.npy')
        self.train_labels = np.load(input_folder + 'cifar_train_labels.npy') '''
        ((self.train_images,self.train_labels),(_,_)) = cifar10.load_data()
        
        ''' load using tensorflow 
        if self.gpu_mode:
            with tf.device('/gpu:0'):
                (self.train_images, self.train_labels), (self.test_images, self.test_labels) = cifar10.load_data()
                self.train_images = self.train_images.reshape(50000, 1024, 3).astype(np.uint8)  # added 3 channels for serialized pixels
                #assert self.train_images.shape == (self.train_images_num, 32, 32, 3)
                self.train_labels = self.train_labels.reshape((50000,))
        else:
            with tf.device('/cpu:0'):
                (self.train_images, self.train_labels), (self.test_images, self.test_labels) = cifar10.load_data()
                self.train_images = self.train_images.reshape(50000, 1024, 3).astype(np.uint8)  # added 3 channels for serialized pixels
                #assert self.train_images.shape == (self.train_images_num, 32, 32, 3)
                self.train_labels = self.train_labels.reshape((50000,)) '''
        
        # `to_categorical` converts this into a matrix with as many
        # columns as there are classes. The number of rows
        # stays the same.
        #self.train_labels = to_categorical(self.train_labels)
        #self.test_labels = to_categorical(self.test_labels)
        
        if self.max_img:
            print ("train_images.shape",self.train_images.shape)
            print ("len(train_labels)",len(self.train_labels))
            #print("train_labels",self.train_labels)
            #print("test_images.shape", self.test_images.shape)
            #print("len(test_labels)", len(self.test_labels))
            #print("test_labels", self.test_labels)
            print('CIFAR data loaded')
        
        
        ''' Create MAD data '''
        ''' This file contains MAD data for each pixel, for each category of image '''
        if self.fit_type[:3] == 'mad':
            try:     # Try to read the MAD data
                f = open(self.in_folder + 'mnist_mad.csv','r')
                data = f.readlines()
                f.close()
                for i in range(len(data)):
                    data[i] = data[i].strip().split(',')
                    for j in range(len(data[i])):
                        data[i][j] = float(data[i][j])
                self.mad = [np.array(data[i]).reshape(self.num_pixels_rgb,) for i in range(len(data))]
            except:    # If no file, create the data and save it to a file
                print('Creating mnist_mad.csv file')
                indices = np.arange(10, dtype=int)
                self.mad = []
                for idx in indices:
                    obs = self.train_images[self.train_labels == idx].reshape(-1, self.num_pixels_rgb).astype(np.float32)/255
                    mad = np.median(obs, axis=0)
                    mad = np.abs(obs-mad)
                    mad = np.median(mad, axis=0)
                    mad = np.minimum(np.maximum(mad, self.min_mad), self.max_mad)
                    self.mad.append(mad)
                del mad
                del obs
                    
                f = open(self.in_folder + 'mnist_mad.csv','w')
                for i in range(len(self.mad)):
                    my_str = ''
                    for j in range(self.mad[i].shape[0]):
                        my_str += str(self.mad[i][j]) + ', '
                    my_str = my_str.rstrip(', ')
                    f.write(my_str + '\n')
                f.close()
                
                '''
                indices = np.arange(10, dtype=int)
                obs = [False for i in range(10)]
                for i in range(len(self.train_labels)):
                    this_dig = np.matmul(self.train_labels[i],indices).astype(int)
                    if isinstance(obs[this_dig], bool):
                        obs[this_dig] = self.train_images[i]
                    else:
                        obs[this_dig] = np.vstack([obs[this_dig], self.train_images[i]])
                        
                # This is a list of pixel medians with one element for each digit type
                self.mad = [np.median(obs[i], axis=0) for i in range(len(obs))] 
                
                # Compute absolute deviations from the pixel medians
                self.mad = [np.abs(np.subtract(obs[i], self.mad[i])) for i in range(len(obs))]
                
                # Compute medians of absolute differences/deviations, that is, MAD
                self.mad = [np.median(self.mad[i], axis=0) for i in range(len(obs))]
                
                # Set minimum value for MAD to avoid division by zero
                self.mad = np.minimum(np.maximum(self.mad, self.min_mad),max_mad)
                
                del obs
                
                f = open(self.in_folder + 'mnist_mad.csv','w')
                for i in range(len(self.mad)):
                    my_str = ''
                    for j in range(self.mad[i].shape[0]):
                        my_str += str(self.mad[i,j]) + ', '
                    my_str = my_str.rstrip(', ')
                    f.write(my_str + '\n')
                f.close()
                '''
        
    def new(self,cifar_idx):
        ''' Define target '''
        self.cifar_idx = cifar_idx
        ''' self.target_img neds to be (3072,) np.float32 numpy array for fitness comptutations '''
        self.target_img = self.train_images[cifar_idx].reshape((3072,)).astype(np.float32)/255
        self.target_label_dig = self.train_labels[cifar_idx]
        
        self.log_filename = 'log_adv_eg_' + self.fit_type + '_' + self.select_type + '_' + self.rand_mode + '_' + str(self.cifar_idx) + '.csv'
        
        ''' Read/Create random population '''
        ''' Code for generating initial populations by random (rand) and mad methods have been commented 
            out because of their inferiority to the brightness (bright) method '''
        ''' Moved code to evolve() '''
        '''
        if self.img_read:
            with open(self.image_folder + (self.img_dump_file % self.cifar_idx), 'rb') as f_in:
                self.pop = np.load(f_in)
                self.pop_size = self.pop.shape[0]
        #elif self.rand_mode == 'bright':
        else:
            self.pop = self.pop_gen_mut() '''
        
        
        '''
        elif self.rand_mode == 'rand':
            self.pop = np.array([self.rand_mnist_w_constraint() for i in range(self.pop_size)], dtype=np.float32)
        elif self.rand_mode == 'mad':
            if self.max_img:
                print('Generating initial population... ', end = '')
            # Create empty population 
            self.pop = np.array([], dtype = np.float32).reshape(0, self.num_pixels, self.num_rgb)
            
            # Iteratively add to population while culling out members with same predicted label as target images 
            while self.pop.shape[0] < self.pop_size:
                pop_add = self.train_images[np.random.randint(0, self.train_images_num - 1, self.pop_size * self.num_pixels).reshape(self.pop_size, self.num_pixels), np.arange(self.num_pixels), :].astype(np.float32)/255
                pop_add_labels = self.check_target_class_pop(pop_add)
                self.pop = np.concatenate((self.pop, pop_add[np.invert(pop_add_labels)]))
            self.pop = self.pop[:self.pop_size]
            if self.max_img:
                print('complete.')
        else:
            if self.max_img:
                print('Infeasible population initializtion.  Population not created.')
        '''
        
        ''' Fitness now computed with fitness() helper function '''
        '''
        if self.fit_type == 'mad':
            self.pop_fit = self.fit_mad()
        elif self.fit_type == 'ssd':
            self.pop_fit = self.fit_ssd()
        else:
            if self.max_img:
                print('Invalid fit_type in GA.__init__')
            exit(1) '''
        
        ''' generate initial population '''
        ''' moved to evolve() '''
        '''
        self.compute_lin_fit_coeff()
        self.fitness()
        self.compute_c_o_prob()
        self.max_fit = 0
        self.get_max_fit(self.max_img)
        if self.max_img:
            print('Random population created')      '''
        
    
    ''' Required for  loading native CIFAR data files '''
    '''
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')  
        return d
    '''

    
    def fitness(self):
        if self.fit_type == 'mad-linear':
            self.pop_fit = self.fit_lin_mad() 
        elif self.fit_type == 'mad-recip':
            self.pop_fit = self.fit_mad()
        elif self.fit_type == 'L2':
            self.pop_fit = self.fit_ssd()
        elif self.fit_type == 'L1':
            self.pop_fit = self.fit_L1()
        elif self.fit_type == 'Linf':
            self.pop_fit = self.fit_L_inf()
        elif self.fit_type == 'L2-lin':
            self.pop_fit = self.fit_ssd_lin()
        elif self.fit_type == 'L1-lin':
            self.pop_fit = self.fit_L1_lin()
        elif self.fit_type == 'Linf-lin':
            self.pop_fit = self.fit_L_inf_lin()
        else:
            print('Invalid fit_type in GA.fitness()')
            exit(1)
        return
    
    def fit_L_inf(self):
        return 1/np.max(np.abs(self.pop.reshape(self.pop_size, self.num_pixels_rgb) - self.target_img), axis=1)
        #return 1/np.max(np.abs(np.subtract(self.pop, self.target_img)), axis=1)

    def fit_L_inf_lin(self):
        return self.Linf_max - np.max(np.abs(self.pop.reshape(self.pop_size, self.num_pixels_rgb) - self.target_img), axis=1) + 0.001
        
    def fit_L1(self):
         return 1/np.sum(np.abs(np.subtract(self.pop.reshape(self.pop_size, self.num_pixels_rgb), self.target_img)), axis=1)
         #return 1/np.sum(np.abs(np.subtract(self.pop, self.target_img)), axis=1)

    def fit_L1_lin(self):
         return self.L1_max - np.sum(np.abs(np.subtract(self.pop.reshape(self.pop_size, self.num_pixels_rgb), self.target_img)), axis=1)

    #def fit_mad(self):
    #    return 1/np.sum(np.abs(self.pop - self.target_img)/self.mad[self.target_label_dig], axis=1)
        #return 1/np.sum(np.divide(np.abs(np.subtract(self.pop, self.target_img)), self.mad[self.target_label_dig]), axis=1)
        
    def fit_mad(self):
        return 1/np.sum(np.divide(np.abs(np.subtract(self.pop.reshape(self.pop_size, self.num_pixels_rgb), self.target_img)), self.mad[self.target_label_dig]), axis=1)
        
    def fit_ssd(self):
        return 1/np.sum((self.pop.reshape(self.pop_size, self.num_pixels_rgb) - self.target_img.reshape(self.num_pixels_rgb))**2, axis = 1)
    
    def fit_lin_mad(self):
        return (self.mad_max - np.sum(np.abs(self.pop.reshape(self.pop_size, self.num_pixels_rgb) - self.target_img)/self.mad[self.target_label_dig], axis=1))
        
    #def fit_ssd(self):
    #    return 1/np.sum((self.pop - self.target_img)**2.0, axis = 1)  #1/np.sum(np.power(np.subtract(self.pop, self.target_img), 2.0), axis = 1)
    
    def fit_ssd_lin(self):
        return self.L2_max - np.sum((self.pop.reshape(self.pop_size, self.num_pixels_rgb) - self.target_img)**2.0, axis = 1)  #1/np.sum(np.power(np.subtract(self.pop, self.target_img), 2.0), axis = 1)
    
    def compute_lin_fit_coeff(self):
        x = np.zeros((1,self.num_pixels_rgb))
        y = np.ones((1,self.num_pixels_rgb))
        z = np.concatenate((x,y))
        bb = np.argmax(np.abs(self.target_img-z), axis = 0)
        
        if self.fit_type == 'mad-linear':
            self.mad_max = np.sum(np.abs(self.target_img-z[bb.reshape(self.num_pixels_rgb), np.arange(self.num_pixels_rgb)])/self.mad[self.target_label_dig])  # .astype(self.fit_parent_dtype)
        else:
            self.mad_max = 0
        if self.fit_type == 'L1-lin':
            self.L1_max = np.sum(np.abs(self.target_img-z[bb.reshape(self.num_pixels_rgb), np.arange(self.num_pixels_rgb)]))
        if self.fit_type == 'L2-lin':
            self.L2_max = np.sum((self.target_img-z[bb.reshape(self.num_pixels_rgb), np.arange(self.num_pixels_rgb)])**2)
        if self.fit_type == 'Linf-lin':
            self.Linf_max = 1.0
     
    #@njit
    def mut_pick(self, idx_mut, npm):
        for i in range(idx_mut.shape[0]):
            idx_mut[i] = np.random.choice(1024, (npm,), replace = False)

    def pop_gen_mut(self):
        ''' Parameters '''
        num_reps = 5000  # Number of mutated images to generate for each number of changed pixels, for each iteration, for each image
        factor = [1.1, 0.8]  # Factors to increase or reduce the amount by which hues are adjusted in mutant pixels
        #num_adv_eg_goal = self.pop_size  # Number o of adv egs to generate per image
        #num_rng = [200, 1667]  # Thresholds for number of adv egs found in each iteration.  If the number of images found
                               # are below the lower threshold then hue adjustment in mutants is increased and above the 
                               # upper threshold the adjustment is reduced
        num_reset_max = 4      # Maximum number of iteration counter resets before the found adv egs are accepted.
                               # This is needed when the neural network misidentifies an image and virtually all 
                               # adv egs are mislabeled.
        num_adv_eg = 0  # counter for number of adv egs found for each image
        num_iter = 0    # counter for iterations for each image
        num_reset = 0   # counter for number of iteration counter resets due to finding an extraordinrily large number of adv egs
        adv_eg = np.empty((0,1024,3)) # initialize repository for adv egs
        # Last scale revision due to removing bias toward darker pixels
        #   New scale values were increased to permit achieving a totally dark/light pixel for a 1-pixel change, consistent
        #     with the prior dark bias 
        # Iteration 2: [1.452   1.089   0.847   0.605   0.605   0.4235  0.363   0.27225 0.1815 ]
        # Iteration 3: [1.5972   1.1979   0.9317   0.6655   0.6655   0.46585  0.3993   0.299475 0.19965 ]
        # [1.8, 1.5, 1.4, 1.0, 1.0, 0.7, 0.6, 0.45, 0.3]
        scale = np.array([1.6, 1.3, 1.1, 0.9, 0.8, 0.7, 0.6, 0.45, 0.3]) #[1.2, 0.9, 0.7, 0.5, 0.5, 0.35, 0.3, 0.225, 0.15]  #[1.2, 0.9, 0.7, 0.5, 0.5, 0.35, 0.3, 0.225, 0.21, 0.175, 0.15, 0.175]
        num_pix_mut = np.array([1, 2, 3, 4, 5, 6, 10, 50, 100]) #[1, 2, 3, 4, 5, 6, 10, 50, 200, 500, 750, 1000]
        self.num_pix_mut_expand = False
        scale_alt = np.array([0.3, 0.3, 0.2])
        num_pix_mut_alt = np.array([200, 500, 1000])
        max_iter = 15
        num_imgs2show = 5
        thres_pop_large = 3
        thres_pop_large_trunc = 3
        
        ''' Convert target images from RGB to HSV and make a copy of it '''
        target_hsv = rgb_to_hsv(self.target_img.reshape(self.num_pixels, self.num_rgb))
        
        
        ''' Continue looping as long as the target number of adv egs has not been achieved or maximum iterations has been reached '''
        while num_adv_eg < self.pop_size and num_iter < max_iter:
            num_iter += 1       # Increment the iteration counter
            print(f'population initialization iteration: {num_iter}; population size: {adv_eg.shape[0]}')
            if num_adv_eg <= 100 and num_iter > 4 and not self.num_pix_mut_expand:
                scale = np.concatenate((scale, scale_alt))
                num_pix_mut = np.concatenate((num_pix_mut, num_pix_mut_alt))
                ''' Set genetic algorithm mutations to strict improvements and more frequent mutations '''
                self.prob_mut_genome = 1.0
                self.prob_mut_pixel = 0.5
                self.mut_light_bias = 0.0
                self.num_pix_mut_expand = True
            num_adv_eg_found = np.zeros(num_pix_mut.shape)
            num_adv_eg_now = 0  # Counter for number of adv egs in current iteration
            for i in range(len(num_pix_mut)):
                idx_mut = np.empty((num_reps,num_pix_mut[i]), dtype = np.uint16)
                x_t_cp = np.tile(target_hsv, (num_reps, 1)).reshape(-1, 1024, 3)
                self.mut_pick(idx_mut, num_pix_mut[i])
                x_t_cp[np.arange(num_reps).reshape(num_reps,1), idx_mut, 2] = np.maximum(np.minimum(x_t_cp[np.arange(num_reps).reshape(num_reps,1), idx_mut, 2] + scale[i] * (np.random.random((num_reps,num_pix_mut[i])) - 0.5), 1.0), 0.0) #pix_mut[np.random.choice(len(pix_mut), size = (50000, npm), replace = True).reshape(50000, npm)]/255
                logits = self.loaded_model.predict(hsv_to_rgb(x_t_cp.reshape(num_reps, 32, 32, 3)))  #.reshape(-1,32,32,3)
                '''
                if self.gpu_mode:
                     with tf.device('/gpu:0'):
                         logits = self.loaded_model.predict(hsv_to_rgb(x_t_cp.reshape(num_reps, 32, 32, 3)))  #.reshape(-1,32,32,3)
                else:
                     with tf.device('/cpu:0'):
                         logits = self.loaded_model.predict(hsv_to_rgb(x_t_cp.reshape(num_reps, 32, 32, 3)))  #.reshape(-1,32,32,3)'''
                if self.max_img:
                    print('Image %d, Iteration %d; Accuracy with %d pixels replaced with scale %f: %f' % (self.cifar_idx, num_iter, num_pix_mut[i], scale[i], (self.target_label_dig == np.argmax(logits, axis=1)).sum()/num_reps))
            
                mismatches = np.where((self.target_label_dig != np.argmax(logits, axis=1)))[0]
                num_adv_eg_found[i] = mismatches.shape[0]
                
                num_adv_eg_now += mismatches.shape[0]
                adv_eg = np.vstack((adv_eg, x_t_cp[mismatches]))
                if self.max_img:
                    if mismatches.shape[0] > 0:
                        print('%d mislabeled' % mismatches.shape[0])
                        for j in mismatches[:num_imgs2show]:
                            fig,ax = plt.subplots(1,2)
                            ax[0].imshow(hsv_to_rgb(target_hsv.reshape(32,32,3)))
                            ax[1].imshow(hsv_to_rgb(x_t_cp[j].reshape(32,32,3)))
                            plt.show()
                else:
                    if self.max_img:
                        print('None mislabeled')
                    
            num_adv_eg += num_adv_eg_now  # Increase number of adv egs found by current iteration total
            if self.max_img:
                print('In %d iterations; Number of adv eg found in this iteration: %d; Total found: %d' % (num_iter, int(num_adv_eg_now), int(num_adv_eg)))
            ''' Write to log file '''
            if self.write_pop_gen:
                with open(self.log_filename, 'a') as f:
                    f.write('%d, %d, %d, %f\n' % (self.cifar_idx, num_iter, num_adv_eg_now, scale[0]))
            
            ''' Adjust brightness mutation factors if number of adv egs found is too few or too many '''
            '''
            if num_adv_eg_now < num_rng[0]:
                scale *= factor[0]
            elif num_adv_eg_now > num_rng[1]:
                scale *= factor[1]
            '''
            scale[num_adv_eg_found > self.pop_size /num_pix_mut.shape[0]] *= factor[1]
            scale[num_adv_eg_found < 0.2 * self.pop_size /num_pix_mut.shape[0]] *= factor[0]
            if self.max_img:
                print(scale)
            
            ''' If found way too many adv egs, then delete adv egs and start again with more tighter mutations '''
            if num_adv_eg_now > thres_pop_large * self.pop_size and num_reset < num_reset_max:
                num_adv_eg = 0
                adv_eg = np.empty((0,1024,3))
                num_iter -= 1
                num_reset += 1
                if self.max_img:
                    print('Reset')
        
        ''' Select the fitness of the adv egs found '''
        ''' If the number of adv egs found is less than the target number, then replicate those that were found '''
        if num_adv_eg > 0:
            adv_eg = hsv_to_rgb(adv_eg)
            
            ''' Dump images to numpy file '''
            if self.img_dump:
                with open(self.in_folder + self.image_folder + self.img_dump_file  % self.cifar_idx, 'wb') as f_dump:
                    np.save(f_dump, adv_eg)
            
            ''' Fitness in RGB encoding '''
            ''' Should HSV encoding be used? '''
            if adv_eg.shape[0] > thres_pop_large_trunc *self.pop_size:
                adv_eg_fit = fitness(hsv_to_rgb(target_hsv), adv_eg)
                idx_best = np.argsort(-adv_eg_fit)[:thres_pop_large_trunc * self.pop_size]
                self.pop_size = idx_best.shape[0]
                return adv_eg[idx_best]
            elif self.init_pop_mode == 'trunc':
                adv_eg_fit = fitness(hsv_to_rgb(target_hsv), adv_eg)
                idx_best = np.argsort(-adv_eg_fit)[:self.pop_size]
                if idx_best.shape[0] < self.pop_size:
                    idx_best = np.tile(idx_best, (int(np.ceil(self.pop_size/idx_best.shape[0])),))
                    idx_best = idx_best[:self.pop_size]
                
                return adv_eg[idx_best] # check shape
            elif self.init_pop_mode == 'all':
                self.pop_size = adv_eg.shape[0]
                return adv_eg
            
        else:
            if self.max_img:
                print('no adv egs found')
        
    
    
    
    
    def rand_mnist_w_constraint(self):
        img =  [self.rand_pixel() for i in range(784)]
        while self.check_target_class(np.array(img)):
            img =  [self.rand_pixel() for i in range(784)]
        return np.array(img)
    
    ''' Function no longer needed after MNIST code was refactored in the conversion to CIFAR code  '''
    '''
    def rand_mnist_mad_w_constraint(self):
        #img =  [self.rand_mad_pixel(i) for i in range(784)]
        while self.check_target_class(np.array(img)):
            img =  [self.rand_mad_pixel(i) for i in range(784)]
        return np.array(img)
    '''
    
    ''' Function not referenced in reamining code '''
    '''
    def rand_pixel(self):
        return 0.0 if random.random() <= self.prob_blk else 1.0 if random.random() <= self.prob_wht else random.randint(1,255)/255
    '''
    
    ''' function not referenced in remaining code '''
    '''
    def rand_mad_pixel(self,i):
        #x = random.random()
        #j = 0
        #while self.f_mad_list[i][j][0] < x:
        #    j +=1
        #return self.f_mad_list[i][j][1]/255
        
        return self.train_images[np.random.randint(0, self.train_images_num), i]
    ''' 
    
    def get_max_fit(self, show_image):
        fit_ind_gen = np.argmax(self.pop_fit)
        self.min_fit = self.pop_fit[np.argmin(self.pop_fit)]
        if self.pop_fit[fit_ind_gen] > self.max_fit:
            self.max_fit = self.pop_fit[fit_ind_gen]
            self.max_fit_img = self.pop[fit_ind_gen]
            if self.max_img:
                print('New best max. fitness: ' + str(self.max_fit))
            if show_image:
                print('Target Image')
                self.show_image(self.target_img.reshape(32,32,3))
                best_fit = np.argmax(self.loaded_model.predict(self.max_fit_img.reshape(-1, 32, 32, 3)))
                '''
                if self.gpu_mode:
                     with tf.device('/gpu:0'):
                         best_fit = np.argmax(self.loaded_model.predict(self.max_fit_img.reshape(-1, 32, 32, 3)))
                else:
                    with tf.device('/cpu:0'):
                         best_fit = np.argmax(self.loaded_model.predict(self.max_fit_img.reshape(-1, 32, 32, 3))) '''
                print('Pop. Best Fit. Classification: ' + str(best_fit), '   Fitness: %f' % (self.max_fit), '   Num. pixels diff.: %d' % ((np.abs(self.target_img.reshape(self.num_pixels,self.num_rgb) - self.pop[fit_ind_gen]) > self.thres_pix_diff).sum(),))
                print('Least fit fitness: %f' % (self.pop_fit[-1],))
                self.show_image(self.max_fit_img)
        else:
            if self.max_img:
                print('No fitness improvement.  Fitness (min, max) = (%f, %f)' % (self.min_fit, self.max_fit))
            
        return
    
    ''' function note needed for data laoded from tensorflow '''
    '''
    def cifar2img(self, imgs):
        if imgs.size == 3072 and imgs.ndim == 1:
            return np.dstack((imgs[:1024], imgs[1024:2048], imgs[2048:])).reshape(32,32,3)
        else:
            imgs = np.dstack((imgs[:,:1024], imgs[:,1024:2048], imgs[:,2048:]))
            return imgs.reshape(-1,32,32,3) '''

    ''' This form of the function was appropriate when data was loaded from native CIFAR data files '''
    '''    
    def show_image(self,img):
            plt.imshow(self.cifar2img(img))
            plt.show()
            return
    '''
    
    def show_image(self,img):
        plt.imshow(img.reshape(32, 32, 3))
        plt.show()
        return
    
    def pick_parents(self, num_needed):
        parents = np.array([np.random.choice(np.arange(self.pop_size), size = self.pop_size, p = self.pop_prob, replace = True), 
               np.random.choice(np.arange(self.pop_size), size = self.pop_size, p = self.pop_prob, replace = True)])
        parents = parents[:, parents[0] != parents[1]]
        while parents[0].shape[0] < num_needed: # self.pop_size:
            parents_new = np.array([np.random.choice(np.arange(self.pop_size), size = self.pop_size, p = self.pop_prob, replace = True), 
               np.random.choice(np.arange(self.pop_size), size = self.pop_size, p = self.pop_prob, replace = True)])
            parents = np. concatenate((parents, parents_new[:, parents_new[0] != parents_new[1]]), axis=1)
        return(parents[:, :num_needed])    # self.pop_size
        
        ''' Old code '''
        '''
        rn = [random.random(), random.random()]
        rn.sort()
        parents = []
        i = 0
        ind = 0
        while len(parents) < 2:
            while rn[ind] > self.pop_cum_prob[i]:
                i += 1
            parents.append(i)
            ind += 1
            if (ind < 2) and (rn[ind] < self.pop_cum_prob[i]):
                parents.append(i)
        return parents
        '''
    
    def compute_c_o_prob(self):
        if self.select_type == 'proportionate':
            self.pop_prob = self.pop_fit / np.sum(self.pop_fit)
        elif self.select_type == 'rank-linear':
            ranks = np.empty((self.pop_size,))
            ranks[np.argsort(self.pop_fit)] = np.arange(self.pop_size)
            self.pop_prob = ranks / np.sum(ranks)
        elif self.select_type == 'rank-nonlinear':
            ranks = np.empty((self.pop_size,))
            ranks[np.argsort(self.pop_fit)] = np.arange(self.pop_size)
            ranks = self.rank_nonlin_factor**(self.pop_size - ranks)
            self.pop_prob = ranks / np.sum(ranks)
        else:
            print('Invalid select_type')
        return
    
    '''
    def mutate_one(self,img):
        if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        img[j] = self.rand_pixel()
        
    def mutate_one_pop(self,pop):
        for i in range(len(pop)):
            if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        pop[i][j] = self.rand_pixel()
    '''
    
    ''' function not referenced in reamaining code '''
    '''
    def mutate_one_mad(self,img):
        if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        img[j] = self.rand_mad_pixel(j)
    '''
        
    def mutate_bright_pop(self,pop):
        
        ''' Transform RGB images to HSV encoding for easy brightness mutation '''
        pop = rgb_to_hsv(pop)
        
        ''' Determine pixels to mutate '''
        mutate_genome_filter = np.random.random((pop.shape[0],)) <= self.prob_mut_genome
        mutate_pixel_filter = np.random.random(pop.shape[:2]) <= self.prob_mut_pixel
        mutate_filter = np.where(mutate_genome_filter[:, np.newaxis] & mutate_pixel_filter)
        
        ''' Determine proportion of brightness gap to close '''
        ''' diminishing step size with generation epoch index '''
        rate = (np.random.random((mutate_filter[0].shape[0],)) - self.mut_light_bias) * (0.985**(self.gen - 1))
        
        ''' Instantiate changes in population '''
        pop[(*mutate_filter,2)] = pop[(*mutate_filter,2)] + (self.target_img.reshape(self.num_pixels,self.num_rgb)[mutate_filter[1],2] - pop[(*mutate_filter,2)]) * rate
        
        ''' Transform HSV images back to RGB '''
        pop = hsv_to_rgb(pop)
        
        ''' Possible improvement is to check for mutations that casued a genome to have the same classification as the target
            and then replace it or alter it until all of the population is feasible.  The advantages of this are that, to some extent
            such a population member is wasted.  On the other hand, it may be an effective parent '''
    
    def mutate_one_mad_pop(self,pop):
        '''
        for i in range(len(pop)):
            if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        pop[i][j] = self.rand_mad_pixel(j)
        '''
        
        
        ''' Determine pixels to mutate '''
        mutate_genome_filter = np.random.random((pop.shape[0],)) <= self.prob_mut_genome
        mutate_pixel_filter = np.random.random(pop.shape[:2]) <= self.prob_mut_pixel
        mutate_filter = np.where(mutate_genome_filter[:, np.newaxis] & mutate_pixel_filter)
        
        ''' Compute mutated pixels '''
        mutants = self.train_images[np.random.randint(0, self.train_images.shape[0]-1, (mutate_filter[1].shape[0],)), mutate_filter[1], :].astype(np.float32)/255
        
        ''' Instantiate mutants '''
        pop[mutate_filter] = mutants
        
        ''' Possible improvement is to check for mutations that casued a genome to have the same calssification as the target
            and then replace it or alter it until all of the population is feasible.  The advantages of this are that, to some extent
            such a population member is wasted.  On the other hand, it may be an effective parent '''
    
    def next_gen_w_contraint(self):
        self.compute_c_o_prob()
        
        ''' find best fit of previous generation '''
        pop_fit_here = [(self.pop_fit[i], i) for i in range(len(self.pop_fit))]
        pop_fit_here.sort(reverse=True)
        best_dim = min(15,len(pop_fit_here))
        sort_ind = [x[1] for x in pop_fit_here[:best_dim]]
        digits = np.argmax(self.loaded_model.predict(self.pop[sort_ind].reshape(-1,32,32,3)), axis=1)
        '''
        if self.gpu_mode:
            with tf.device('/gpu:0'):
                digits = np.argmax(self.loaded_model.predict(self.pop[sort_ind].reshape(-1,32,32,3)), axis=1)
        else:
            with tf.device('/cpu:0'):
                digits = np.argmax(self.loaded_model.predict(self.pop[sort_ind].reshape(-1,32,32,3)), axis=1) '''
        
        #digits = [(np.argmax(self.loaded_model.predict(np.array([self.pop[pop_fit_here[i][1]].reshape(784,)]))),i) for i in range(best_dim)]
        ''' find best fit for each category '''
        ''' also, count how many categories are occupied by best fit from above'''
        best_ind = np.array([-1 for i in range(10)])
        for i in range(len(digits)-1,-1,-1):
            best_ind[digits[i]] = pop_fit_here[i][1]
        # Safeguard in case any of the population have been mutated into the target class
        best_ind[self.target_label_dig] = -1
        #count_good = sum([1 for x in best_ind if x >= 0])
        count_good = np.sum(best_ind >= 0)
        
        # adjusted statement below for numpy parents data type
        #parents = [tuple(self.pick_parents()) for i in range(self.pop_size - count_good)]
        parents = self.pick_parents(self.pop_size - count_good).T
        crossover = np.random.randint(0,self.num_pixels - 1, size = (self.pop_size - count_good,))
        new_pop = [[np.concatenate([self.pop[parents[i][0]][:crossover[i], :], self.pop[parents[i][1]][crossover[i]:, :]])] for i in range(self.pop_size - count_good)]
        new_pop = np.vstack(new_pop).astype(np.float32)
        
        ''' Mutation '''
        ''' The mutation method formerly associated with self.rand_mode=='bright' is now used for all mutations '''
        '''
        if self.rand_mode == 'rand':
            self.mutate_one_pop(new_pop)
        elif self.rand_mode == 'mad':
            self.mutate_one_mad_pop(new_pop)
        elif self.rand_mode == 'bright':
            self.mutate_bright_pop(new_pop) '''
            
        self.mutate_bright_pop(new_pop)
        
        #new_pop = np.array(new_pop, dtype = np.float32)        
        target_match = self.check_target_class_pop(new_pop)
        random_select = np.where(np.random.random((target_match.sum(),)) <= 0.5, 0, 1)
        
        new_pop[target_match] = self.pop[parents[np.where(target_match)[0], random_select]]
        
        for i in range(len(best_ind)):
            if best_ind[i] >= 0:
                new_pop = np.vstack((new_pop, self.pop[best_ind[i]][np.newaxis,:,:]))
        
        '''
        for i in range(len(new_pop)):
            if target_match[i]:
                if random.random() <=0.5:
                    new_pop[i] = parents[i][0]
                else:
                    new_pop[i] = parents[i][1]
        '''
        
                
        self.pop = new_pop.copy()
        
        ''' Fitness now computed with fitness() helper() function '''
        '''
        if self.fit_type == 'mad':
            self.pop_fit = self.fit_mad() #[self.fitness_mad(p) for p in self.pop]
        else:
            self.pop_fit = self.fit_ssd() #[self.fitness(p) for p in self.pop]
        '''
        
        self.get_max_fit(self.max_img) 
        self.fitness()
        
        return
        
    
    def evolve(self):
        start_time = datetime.datetime.now()
        
        ''' Generate initial population '''
        try:
            with open(self.in_folder + (self.img_dump_file % self.cifar_idx), 'rb') as f_in:
                self.pop = np.load(f_in)
                self.pop_size = self.pop.shape[0]
        #elif self.rand_mode == 'bright':
        except:
            self.pop = self.pop_gen_mut()
            with open(self.in_folder + 'pop_' + str(self.cifar_idx) + '.npy', 'wb') as f:
                np.save(f, self.pop)
        print('population initialized')
        
        self.compute_lin_fit_coeff()
        self.fitness()
        self.compute_c_o_prob()
        self.max_fit = 0
        self.get_max_fit(self.max_img)
        time_pop_init = datetime.datetime.now() - start_time
        if self.max_img:
            print('Random population created')
        
        for i in range(self.num_gen):
            self.gen = i
            print(f'Generation {i}')
            if self.max_img:
                print('Generation ' + str(i+1) + ':  ', end='')
            self.next_gen_w_contraint()
            
            ''' Write Results to File '''
            #f_out.write(str(i) + ',' + str(self.max_fit) + ',' + str(np.mean(self.pop_fit)) + '\n')
        
        ''' Write Best Fit Results to File'''
        predict = np.argmax(self.loaded_model.predict(self.pop.reshape(-1, 32, 32, 3)), axis = 1)
        '''
        if self.gpu_mode:
            with tf.device('/gpu:0'):
                predict = np.argmax(self.loaded_model.predict(self.pop.reshape(-1, 32, 32, 3)), axis = 1)
        else:
            with tf.device('/cpu:0'):
                predict = np.argmax(self.loaded_model.predict(self.pop.reshape(-1, 32, 32, 3)), axis = 1) '''
        combine_fit_pop = [(self.pop_fit[i], self.pop[i], predict[i]) for i in range(self.pop_size)]
        combine_fit_pop.sort(reverse=True, key = lambda x:x[0])
        
        ''' Get best images, fitness, and label and initiate numpy arrays with those elements '''
        done = False
        max_fit = combine_fit_pop[0][0]
        digit_done = [False for i in range(10)]
        best_images = np.array([combine_fit_pop[0][1]])
        digit_done[combine_fit_pop[0][2]] = True
        i = 1
        best_labels = [combine_fit_pop[0][2]]
        best_images_fit = [combine_fit_pop[0][0]]
        
        ''' If desired collect best fitness adv eg of distinct predicted labels '''
        if not self.out_only_best:
            while i <= self.pop_size - 1 and not done:
                if combine_fit_pop[i][0] > 0.90 * max_fit:
                    if digit_done[combine_fit_pop[i][2]] == False:
                        digit_done[combine_fit_pop[i][2]] = True
                        best_images = np.vstack([best_images,combine_fit_pop[i][1][np.newaxis,:,:]])
                        best_labels.append(combine_fit_pop[i][2])
                        best_images_fit.append(combine_fit_pop[i][0])
                    i += 1
                else:
                    done = True
                
        '''
        f = open('./output/images/' + self.filename + '_fit.csv','w')
        for i in range(1,15):
            f.write(str(combine_fit_pop[i][0]) + '\n')
            best_images = np.vstack([best_images,combine_fit_pop[i][1]])
        f.close()
        np.savetxt('./output/images/' + self.filename + '_images.csv', best_images, delimiter=",",fmt='%8.6f',newline='\n')
        '''
        
        #finish_time = datetime.datetime.now()
        
        ''' Report Results to console '''
        if self.max_img:
            print('\n\n\n\nResults')
            print('Maximum fitness: ' + str(self.max_fit) + '\n')
            print('Target Image')
            self.show_image(self.target_img)
                
            print('Best Fit Images')
            #print('Best Fit Classification: ' + str(np.argmax(self.loaded_model.predict(np.array([self.max_fit_img])))))
            #self.show_image(self.max_fit_img)
            for i in range(len(best_images)):
                print('Image label: ' + str(best_labels[i]), '   Maximum fitness: ' + str(self.max_fit) + '\n', '   Num. pixels diff.: %d' % ((np.abs(self.target_img.reshape(self.num_pixels,self.num_rgb) - best_images[0]) > self.thres_pix_diff).sum(),))
                self.show_image(best_images[i])
        
        
        ''' write best image to file '''
        if self.write_adv_eg:
            with open(self.out_folder + self.image_folder + (self.best_fit_file % (self.cifar_idx)), 'wb') as f_best:
                np.save(f_best, best_images[0])
        
        ''' Put best into database '''
        '''
        cnx = MySQL.connect(user='root', passwd='MySQL', host='127.0.0.1', db='adv_exmpl')
        cursor = cnx.cursor(buffered=True)
        for i in range(len(best_images)):
            img_str = ''
            for j in range(best_images[i].shape[0]):
                img_str += str(best_images[i][j]) + ' '
            cursor.callproc('spInsertRow',(self.cifar_idx,int(self.target_label_dig),int(best_labels[i]), img_str)) # best_images[i].tobytes()
            cnx.commit() '''
            
        ''' Create output '''
        finish_time = datetime.datetime.now()
        elapse_time = finish_time - start_time
        #f_out = open(self.out_folder + self.log_name + '.csv','a')
        #log_name_stripped = self.re_strip.sub('',self.log_name)
        output = []
        if self.out_only_best:
            img_str = ' '.join([str(i) for i in best_images[0].flatten()])
            output.append(str(self.cifar_idx) + ', ' + str(self.target_label_dig) + ', ' + str(best_labels[0]) + ', ' + str(best_images_fit[0]) + ',' + str(time_pop_init) + ',' + str(elapse_time) + ', ' + img_str + '\n')
        else:
            for i in range(len(best_images)):
                #img_str = '"'
                img_str = ''
                ''' Code revised to output only best adversarial example '''
                '''
                for j in range(best_images[i].shape[0]):
                    img_str += str(best_images[i][j]) + ' ' '''
                img_str = ' '.join([str(i) for i in best_images[i].flatten()])
                output.append(str(self.cifar_idx) + ', ' + str(self.target_label_dig) + ', ' + str(best_labels[i]) + ', ' + str(best_images_fit[i]) + ',' + str(time_pop_init) + ',' + str(elapse_time) + ', ' + img_str + '\n')
                #img_str += '"'
                #out_str = log_name_stripped  + ', ' + str(self.cifar_idx) + ', ' + str(self.target_label_dig) + ', ' + str(best_labels[i]) + ', ' + str(self.max_fit) + ', ' + img_str + '\n'
                #f_out.write(out_str)
            #f_out.close()
        
        ''' Write to log file '''
        #finish_time = datetime.datetime.now()
        #elapse_time = finish_time - start_time
        #minutes = int(elapse_time.total_seconds()/60)
        #seconds = elapse_time.total_seconds() - minutes * 60.0
        #f_out = open(self.out_folder + self.log_name + '.log','a')
        #f_out.write('MNIST index ' + str(self.cifar_idx) + ' completed at ' + datetime.datetime.strftime(datetime.datetime.now(), '%m/%d/%y %H:%M:%S') + ' in ' + str(minutes) + ':' + str(seconds) + ' with fitness ' + str(self.max_fit) + '\n')
        #f_out.close()
        
        #new_log_name = re.sub('E\d+', 'E' + str(self.cifar_idx), self.log_name)
        
        #os.rename(self.out_folder + self.log_name + '.csv', self.out_folder + new_log_name + '.csv')
        #os.rename(self.out_folder + self.log_name + '.log', self.out_folder + new_log_name + '.log')
        
        #self.log_name = new_log_name
        
        return output
        
    ''' function not referenced in remaining code '''
    '''
    def mutate(self):
        for i in range(len(self.pop)):
            if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        self.pop[i,j] = self.rand_pixel()
        return
    '''
    
    ''' function not referenced in remaining code '''
    '''
    def crossover(self):
        keep_best_fit = True
        new_pop = []
        for i in range(len(self.pop)):
            p1, p2 = self.pick_parents()
            cross_pt = random.randint(0,len(self.pop) - 1)
            new_pop.append(np.concatenate([self.pop[p1][0:cross_pt], self.pop[p2][cross_pt:]]))
            
        self.pop = np.array([p for p in new_pop])
        
        if keep_best_fit:
            self.pop = np.vstack([self.pop[0:len(new_pop)-1], self.max_fit_img])
        return 
    '''

    def check_target_class(self, img):
        # assumes img.shape = (32,32,3)
        #return np.argmax(self.loaded_model.predict(np.array([img]), batch_size=1000)) == self.target_label_dig
        try:
            assert isinstance(img, np.ndarray) and img.shape[0] == 1
        except:
            print('Data sent to check_target_class(self, img) was not a numpy array containing one CIFAR image')
            exit(1)
        
        result = np.argmax(self.loaded_model.predict(img.reshape(1,32,32,3))) == self.target_label_dig
        '''
        if self.gpu_mode:
            with tf.device('/gpu:0'):
                result = np.argmax(self.loaded_model.predict(img.reshape(1,32,32,3))) == self.target_label_dig
        else:
            with tf.device('/cpu:0'):
                result = np.argmax(self.loaded_model.predict(img.reshape(1,32,32,3))) == self.target_label_dig '''
        return result
    
    def check_target_class_pop(self, pop):
        # Assumes prediction of more than one image
        result = np.argmax(self.loaded_model.predict(pop.reshape(-1,32,32,3)), axis = 1) == self.target_label_dig
        '''
        if self.gpu_mode:
            with tf.device('/gpu:0'):
                result = np.argmax(self.loaded_model.predict(pop.reshape(-1,32,32,3)), axis = 1) == self.target_label_dig
        else:
            with tf.device('/cpu:0'):
                result = np.argmax(self.loaded_model.predict(pop.reshape(-1,32,32,3)), axis = 1) == self.target_label_dig'''
        return result
    
""" Function Definitions """

""" Create random grayscale images with pixels on [0.0, 1.0] """
def randDig():
    blkProb = 0.7
    whtProb = 0.25/(1 - blkProb)
    #img =  [[ 0.0 if random.random() <= blkProb else 1.0 if random.random() <= whtProb else random.randint(1,255)/255 for j in range(28)] for i in range(28)]
    img =  [0.0 if random.random() <= blkProb else 1.0 if random.random() <= whtProb else random.randint(1,255)/255 for i in range(784)]
    return np.array(img)

"""
def fitness(bench,cand):
    ssq = 0.0
    for i in range(len(bench)):
        ssq += (bench[i] - cand[i])^2
    return 1/ssq"""
    
def fitness(target,cand):
    return 1/np.sum((target - cand)**2, axis = (1,2))



''' Handle input arguments '''
parser = argparse.ArgumentParser(description='Generate adversarial examples for neural network')
parser.add_argument('cifar_id', metavar='cifar_id', type=int, help='CIFAR target index')
parser.add_argument('model_file', metavar='model_file', type=str, help='JSON file for neural network model')
#parser.add_argument('file_weights', metavar='file_weights', type=str, help='h5 file for neural network weights')
parser.add_argument('out_folder', metavar='out_folder', type=str, help='file folder for output')
parser.add_argument('folder', metavar='folder', type=str, help='base file folder for code/input/output subfolders')
parser.add_argument('fit_type', metavar='fit_type', type=str, help='Fitness function type')
parser.add_argument('select_type', metavar='select_type', type=str, help='Selection method for parents')
''' Note: args.rand_type is not used as all mutations and initial populations are generated with the formerly called "bright" mode '''
parser.add_argument('rand_type', metavar='rand_type', type=str, help='Randomization mode for mutation and initializing populations: rand = uniformly random pixel values on [0.0,1.1]; mad = drawn from observed distribution of pixel values')
parser.add_argument('factor_rank_nonlinear', metavar='factor_rank_nonlinear', type=float, help='Factor for nonlinear rank selectio')
parser.add_argument('gpu_mode', metavar='gpu_mode', type=str, help='GPU/CPU prediction')
args = parser.parse_args()


''' Hide GPU from tensorflow '''
if not str_to_bool(args.gpu_mode):
    print('Hello: GPU is hidden')
    tf.config.experimental.set_visible_devices([], 'GPU')


''' Set GA parameters '''
pop_size = 2000   # population size
prob_mut_genome = .4 #1.0  # probability of mutation
prob_mut_pixel = 0.25  #0.5 
mut_light_bias = 0.1
num_gen = 125     # number of generations
max_fit = 0       # maximum fitness: initialize to 0
pop_fit = []      # population fitness
model_filename = args.model_file #'ff_mnist.json'
min_mad = 0.1    # Values tried: 0.001, 0.05
max_mad = 0.5 #0.15
''' Note: args.rand_type is not used as all mutations and initial populations are generated with the formerly called "bright" mode '''
rand_type = 'bright'
out_only_best = True
input_folder = args.folder #+  'data/' #'/sciclone/home10/jrbrad/files/mnist/input/'
output_folder = args.out_folder #'/sciclone/home10/jrbrad/files/mnist/output/'
prints_img = False
print(f'output folder: {args.out_folder}')
        

'''
if os.environ['COMPUTERNAME'] == 'BRADLEYJ-5810':
    input_folder = 'D:/research/neuralnetworks/code/mnist/adversarial_eg/ga/input/'
    output_folder = 'D:/Research/NeuralNetworks/code/MNIST/adversarial_eg/ga/cl/output/'
    prints_img = True
else:
    input_folder = '/sciclone/home10/jrbrad/files/mnist/input/'
    output_folder = '/sciclone/home10/jrbrad/files/mnist/output/'
    prints_img = False
'''

    
#log_name = re.sub('_','-',re.sub('\.json','',model_filename)) + '_' + re.sub('_','-',re.sub('\.h5','',weights_filename)) + '_' + 'pop' + str(pop_size) + '_' + 'mutgenome' + str(int(prob_mut_genome*100)) + '_' + 'mutpix' + str(int(prob_mut_pixel*1000)) + '_' + 'gen' + str(num_gen) + '_' + 'fit-' + fit_type + '_' + 'rand-' + rand_type + '_B' + str(args.start) + 'E' + str(args.start)
''' Create empty output file '''
#f_out = open(output_folder + log_name + '.csv','w')
#f_out.close()
''' Create empty log file '''
#f_out = open(output_folder + log_name + '.log','w')
#f_out.close()

scen_name = re.sub('_','-',re.sub('\.json','',model_filename)) + '_' + 'pop' + str(pop_size) + '_' + 'mutgenome' + str(int(prob_mut_genome*100)) + '_' + 'mutpix' + str(int(prob_mut_pixel*1000)) + '_' + 'gen' + str(num_gen) + '_' + 'fit-' + args.fit_type + '_' + 'rand-' + rand_type
#  + '_' + re.sub('_','-',re.sub('\.h5','',weights_filename))

''' Instantiate GA object '''
print(f'GA args: {pop_size, num_gen, prob_mut_genome, prob_mut_pixel, mut_light_bias, num_gen, args.folder, input_folder, model_filename, output_folder, prints_img, args.fit_type, args.select_type, args.factor_rank_nonlinear, min_mad, max_mad, rand_type, args.gpu_mode, out_only_best}')
#ga = GA(pop_size, num_gen, prob_mut_genome, prob_mut_pixel, prob_wht, prob_blk, num_gen, i, input_folder, output_folder, loaded_model, prints_img, log_name, fit_type, min_mad, 'mad')
ga = GA(pop_size, num_gen, prob_mut_genome, prob_mut_pixel, mut_light_bias, num_gen, args.folder, input_folder, model_filename, output_folder, 
        prints_img, args.fit_type, args.select_type, args.factor_rank_nonlinear, min_mad, max_mad, rand_type, args.gpu_mode, out_only_best)
print('ga initialized')
ga.new(args.cifar_id)
print('ga.new() complete')
result = ga.evolve()
for i in range(len(result)):
    result[i] = scen_name + ',' + result[i]
print(result, end='')