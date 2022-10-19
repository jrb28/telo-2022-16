# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:08:12 2020

@author: jrbrad
"""

'''
Copyright 2022 by Anonymous Authors
This code is protected by the Academic Free license Version 3.0
See LICENSE.md in the repository root folder for the terms of this license
'''

#import matplotlib.pyplot as plt
import random
#import glob

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras import models
from keras.utils import to_categorical
from keras.datasets import mnist

import numpy as np
import json
import argparse
import os
import datetime
import time
import re

def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

class GA():
    
    def new(self,mnist_index):
        
        max_mad = 0.15
        
        ''' Load and pre-treat MNIST data '''
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        #Reshape to 60000 x 784
        self.train_images = self.train_images.reshape((60000, 28 * 28))
        self.test_images = self.test_images.reshape((10000, 28 * 28))
        
        # Normalize range of data to 0,1
        self.train_images = self.train_images / 255  #.astype(self.fit_parent_dtype)
        self.test_images = self.test_images / 255  # .astype(self.fit_parent_dtype)
        
        # `to_categorical` converts this into a matrix with as many
        # columns as there are classes. The number of rows
        # stays the same.
        #self.mnist_index = target_index
        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)
        #self.target_img = self.train_images[target_index]  # numpy array of length 784
        #self.target_label = self.train_labels[target_index] # numpy array of length 10, on-hot encoded
        #self.target_label_dig = np.argmax(self.target_label)
        
        if self.max_img:
            print ("train_images.shape",self.train_images.shape)
            print ("len(train_labels)",len(self.train_labels))
            print("train_labels",self.train_labels)
            print("test_images.shape", self.test_images.shape)
            print("len(test_labels)", len(self.test_labels))
            print("test_labels", self.test_labels)
    
            print('MNIST data loaded and pre-conditioned')
        
        
        ''' Create MAD data '''
        if self.fit_type[:3] == 'mad':
            try:     # Try to read the MAD data
                f = open(self.in_folder + 'mnist_mad.csv','r')
                data = f.readlines()
                f.close()
                for i in range(len(data)):
                    data[i] = data[i].strip().split(',')
                    for j in range(len(data[i])):
                        data[i][j] = float(data[i][j])
                self.mad = [np.array(data[i]).reshape(784,) for i in range(len(data))]
                self.mad = np.array(self.mad)     # .astype(self.fit_parent_dtype)
            except:    # If no file, create the data and save it to a file
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
                self.mad = np.array(self.mad)    #.astype(self.fit_parent_dtype)
                
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
        
        
        
        
        
        ''' Define target '''
        self.mnist_index = mnist_index
        self.target_img = self.train_images[mnist_index].astype(self.fit_parent_dtype)  # numpy array of length 784
        self.num_pixels = self.target_img.shape[0]
        self.target_label = self.train_labels[mnist_index] # numpy array of length 10, on-hot encoded
        self.target_label_dig = np.argmax(self.target_label)
        
        del self.train_images
        del self.train_labels
        del self.test_images
        del self.test_labels
        
        ''' Create random population '''
        if self.rand_mode == 'rand':
            self.pop = np.array([self.rand_mnist_w_constraint() for i in range(self.pop_size)], dtype=self.fit_parent_dtype)
        elif self.rand_mode == 'mad':
            with open(self.in_folder + 'mad_dist.json', 'r') as f:
                f_mad_dict = json.load(f, object_hook=jsonKeys2int)
            self.f_mad_list = []
            for i in range(len(f_mad_dict)):
                self.f_mad_list.append([(v,k) for k,v in f_mad_dict[i].items()])
            del f_mad_dict
            self.pop = np.array([self.rand_mnist_mad_w_constraint() for i in range(self.pop_size)], dtype=self.fit_parent_dtype)
        else:
            if self.max_img:
                print('Infeasible population initializtion.  Population not created.')
        
        '''
        if self.fit_type == 'mad-linear':
            self.mad_max = self.get_mad_max()
            self.pop_fit = self.fit_lin_mad() #self.fit_lin_mad()
        elif self.fit_type == 'mad-recip':
            self.pop_fit = self.fit_mad()
        elif self.fit_type == 'L2':
            self.pop_fit = self.fit_ssd()
        elif self.fit_type == 'L1':
            self.pop_fit = self.fit_L1()
        elif self.fit_type == 'Linf':
            self.pop_fit = self.fit_L_inf()
        else:
            print('Invalid fit_type in GA.__init__')
            exit(1)
        '''
        
        self.compute_lin_fit_coeff()
        '''
        if self.fit_type == 'mad-linear':
            self.mad_max = self.get_mad_max()
        else:
            self.mad_max = 0 '''
        self.fitness()
        #self.compute_c_o_prob()
        self.max_fit = 0
        self.get_max_fit(self.max_img)
        if self.max_img:
            print('Random population created')      
        
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
        return 1/np.max(np.abs(self.pop - self.target_img), axis=1)
        #return 1/np.max(np.abs(np.subtract(self.pop, self.target_img)), axis=1)

    def fit_L_inf_lin(self):
        return self.Linf_max - np.max(np.abs(self.pop - self.target_img), axis=1) + 0.001
        
    def fit_L1(self):
         return 1/np.sum(np.abs(np.subtract(self.pop, self.target_img)), axis=1)
         #return 1/np.sum(np.abs(np.subtract(self.pop, self.target_img)), axis=1)

    def fit_L1_lin(self):
         return self.L1_max - np.sum(np.abs(np.subtract(self.pop, self.target_img)), axis=1)

    def fit_mad(self):
        return 1/np.sum(np.abs(self.pop - self.target_img)/self.mad[self.target_label_dig], axis=1)
        #return 1/np.sum(np.divide(np.abs(np.subtract(self.pop, self.target_img)), self.mad[self.target_label_dig]), axis=1)
        
    ''' function replaced by the one below, which computes maximuma for all linear fitness functions 
    def get_mad_max(self):
        x = np.zeros((1,self.num_pixels))
        y = np.ones((1,self.num_pixels))
        z = np.concatenate((x,y))
        bb = np.argmax(np.abs(self.target_img-z), axis = 0)
        
        return np.sum(np.abs(self.target_img-z[bb.reshape(self.num_pixels), np.arange(self.num_pixels)])/self.mad[self.target_label_dig])  # .astype(self.fit_parent_dtype) '''
        
    def compute_lin_fit_coeff(self):
        x = np.zeros((1,self.num_pixels))
        y = np.ones((1,self.num_pixels))
        z = np.concatenate((x,y))
        bb = np.argmax(np.abs(self.target_img-z), axis = 0)
        
        if self.fit_type == 'mad-linear':
            self.mad_max = np.sum(np.abs(self.target_img-z[bb.reshape(self.num_pixels), np.arange(self.num_pixels)])/self.mad[self.target_label_dig])  # .astype(self.fit_parent_dtype)
        else:
            self.mad_max = 0
        if self.fit_type == 'L1-lin':
            self.L1_max = np.sum(np.abs(self.target_img-z[bb.reshape(self.num_pixels), np.arange(self.num_pixels)]))
        if self.fit_type == 'L2-lin':
            self.L2_max = np.sum((self.target_img-z[bb.reshape(self.num_pixels), np.arange(self.num_pixels)])**2)
        if self.fit_type == 'Linf-lin':
            self.Linf_max = 1.0
        
        
    def fit_lin_mad(self):
        return (self.mad_max - np.sum(np.abs(self.pop - self.target_img)/self.mad[self.target_label_dig], axis=1))
        
    ''' fit_rank() not used 
    def fit_rank(self):
        fitness = self.mad_max - np.sum(np.abs(self.pop - self.target_img)/self.mad[self.target_label_dig], axis=1)
        ranks = np.empty((self.pop_size,))
        ranks[np.argsort(fitness)] = np.arange(self.pop_size).astype(np.int16)
        print('Max fitness: %f;  Average fitness: %f' % (float(fitness.max()), float(sum(fitness)/self.pop_size)))
        #return 0.1**(self.pop_size - ranks)
        return ranks '''
        
        
    def fit_ssd(self):
        return 1/np.sum((self.pop - self.target_img)**2.0, axis = 1)  #1/np.sum(np.power(np.subtract(self.pop, self.target_img), 2.0), axis = 1)
    
    def fit_ssd_lin(self):
        return self.L2_max - np.sum((self.pop - self.target_img)**2.0, axis = 1)  #1/np.sum(np.power(np.subtract(self.pop, self.target_img), 2.0), axis = 1)
    
    #def __init__(self, pop_size, num_gen, prob_mut_genome, prob_mut_pixel, prob_wht, prob_blk, gen, target_index, input_folder, output_folder, loaded_model, max_img, log_name, fit_type = 'mad', min_mad = 0.1, rand_mode = 'rand'):
    def __init__(self, pop_size, num_gen, prob_mut_genome, prob_mut_pixel, 
                 prob_wht, prob_blk, gen, input_folder, model_filename, weights_filename, 
                 output_folder, max_img, fit_type, select_type, min_mad, rand_mode, factor_rank_nonlinear, 
                 scen_name):
        ''' Set general parameters '''
        self.fit_parent_dtype = np.float64
        self.pop_size = pop_size
        self.num_gen = num_gen
        self.min_mad = self.fit_parent_dtype(min_mad)
        self.fit_type = fit_type
        self.rand_mode = rand_mode
        self.select_type = select_type
        self.rank_nonlin_factor = factor_rank_nonlinear #0.25 #0.1
        #self.rand_mode = rand_mode
        self.in_folder = input_folder
        self.max_img = max_img
        self.scen_name = scen_name +'/'
        #self.out_folder = output_folder
        self.out_folder = output_folder + self.scen_name
        #self.log_name = log_name
        ''' re object for stripping filename characters '''
        self.re_strip = re.compile('_B\d+E\d+')
        
        ''' Create folder for scenario (scen_name) output files if it doesn't exist '''
        ''' The code below doesn't work in parallel sometimes (hypothesized) because multiple threads test true in the if statement
            and before the directory can be established, another process creates the directory '''
        '''
        if not os.path.isdir(output_folder + self.scen_name):
            os.mkdir(output_folder + self.scen_name) '''
            
        ''' The following code to establish an output folder was moved to the ga_control code file'''
        '''
        try:
            os.mkdir(self.out_folder)
        except:
            pass '''
        
        ''' Load Model & Weights, and Compile '''
        json_file = open(self.in_folder + model_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_abbrv = model_filename.replace('.json','')
        self.loaded_model = models.model_from_json(loaded_model_json)
        self.loaded_model.load_weights(self.in_folder + weights_filename)
        self.loaded_model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        if self.max_img:
            print("Model loaded from disk and compiled")

        
        ''' Load Model and Compile '''
        #self.loaded_model = loaded_model
        '''
        json_file = open(self.in_folder + model_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = models.model_from_json(loaded_model_json)
        #del loaded_model_json
        # load weights into new model
        self.loaded_model.load_weights(self.in_folder + weights_filename)
        
        #Compile
        self.loaded_model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        print("Model loaded from disk and compiled")
        '''
        
        ''' Set random population parameters'''
        self.prob_mut_genome = prob_mut_genome
        self.prob_mut_pixel = prob_mut_pixel
        self.prob_wht = prob_wht
        self.prob_blk = prob_blk
        '''
        if rand_mode == 'rand':
            self.pop = np.array([self.rand_mnist_w_constraint() for i in range(pop_size)], dtype='float32')
        elif rand_mode == 'mad':
            with open(self.in_folder + 'mad_dist.json', 'r') as f:
                f_mad_dict = json.load(f, object_hook=jsonKeys2int)
            self.f_mad_list = []
            for i in range(len(f_mad_dict)):
                self.f_mad_list.append([(v,k) for k,v in f_mad_dict[i].items()])
            del f_mad_dict
            self.pop = np.array([self.rand_mnist_mad_w_constraint() for i in range(pop_size)], dtype='float32')
        else:
            print('Infeasible population initializtion.  Population not created.')
        if self.fit_type == 'mad':
            #self.pop_fit = [self.fitness_mad(p) for p in self.pop]
            #self.pop_fit = 1/np.sum(np.divide(np.abs(np.subtract(self.pop, self.target_img)), self.mad[self.target_label_dig]), axis=1)
            self.pop_fit = self.fit_mad()
        elif self.fit_type == 'ssd':
            #self.pop_fit = [self.fitness(p) for p in self.pop]
            #self.pop_fit = 1/np.sum(np.power(np.subtract(self.pop, self.target_img), 2.0), axis = 1)
            self.pop_fit = self.fit_ssd()
        else:
            print('Invalid fit_type in GA.__init__')
            exit(1)
        
        self.max_fit = 0
        self.get_max_fit(self.max_img)
        if self.max_img:
            print('Random population created')
        '''
        
        # Create output file
        # Filename is  the label_index_numgen_popsize_probWht_probBlk_probMutGenome(x100)_probMutPixel(x1000)_fileIndex
        #self.filename = str(self.target_label_dig) + '_' + str(self.num_gen) + '_' + str(self.pop_size) + '_' + str(int(self.prob_wht * 100)) + '_' + str(int(self.prob_blk * 1000)) + '_' + 'constraint' + '_' + self.fit_type
        #files = glob.glob('./output/' + self.filename  + '*')
        #self.filename += '_' + str(len(files)) #+ '.csv'
        
    ''' Function no longer used 
    def pop_condition(self):
        # condition population to  remove images classified same as benchmark
        old_changed = range(len(self.pop))
        while len(old_changed):
            new_changed = []
            for i in old_changed:
                if np.argmax(self.loaded_model.predict(np.array([self.pop[i]]))) == self.target_label_dig:
                    self.pop[i] = np.array(self.rand_mnist())
                    new_changed.append(i)
            old_changed = [x for x in new_changed]
        return '''

    ''' Function no longer used 
    def rand_mnist(self):
        #blkProb = self.prob_blk
        #whtProb = 0.25/(1 - self.prob_blk)
        #img =  [[ 0.0 if random.random() <= blkProb else 1.0 if random.random() <= whtProb else random.randint(1,255)/255 for j in range(28)] for i in range(28)]
        #img =  [0.0 if random.random() <= blkProb else 1.0 if random.random() <= whtProb else random.randint(1,255)/255 for i in range(784)]
        #img =  [self.rand_pixel() for i in range(784)]
        #img =  [random.random() for i in range(784)]
        img =  np.random.rand(self.num_pixels)
        #return np.array(img)
        return img '''
    
    ''' function revised for  simple uniform random value on [0,1] (see below)
    def rand_mnist_w_constraint(self):
        img =  [self.rand_pixel() for i in range(784)]
        while self.check_target_class(np.array(img)):
            img =  [self.rand_pixel() for i in range(784)]
        return np.array(img) '''
    
    def rand_mnist_w_constraint(self):
        img =  np.random.rand(self.num_pixels)
        while self.check_target_class(img):
            img =  np.random.rand(self.num_pixels)
        return img
    
    def rand_mnist_mad_w_constraint(self):
        img =  [self.rand_mad_pixel(i) for i in range(self.num_pixels)]
        while self.check_target_class(np.array(img)):
            img =  [self.rand_mad_pixel(i) for i in range(self.num_pixels)]
        return np.array(img)
    
    ''' Current instantiation mutates pixels from uniform distribution on [0,1] '''
    '''
    def rand_pixel(self):
        return 0.0 if random.random() <= self.prob_blk else 1.0 if random.random() <= self.prob_wht else random.randint(1,255)/255 '''
    
    def rand_mad_pixel(self,i):
        x = random.random()
        j = 0
        while self.f_mad_list[i][j][0] < x:
            j +=1
        return self.f_mad_list[i][j][1]/255
     
    '''
    def fitness(self,cand):
        return 1/np.sum((self.target_img - cand)**2)
    
    def fitness_mad(self, img):
        return 1/np.sum(np.divide(np.abs(np.subtract(self.target_img, img)),self.mad[self.target_label_dig]))
    '''
    
    def get_max_fit(self, show_image):
        #max_fit_gen = 0
        #fit_ind_gen = -1
        
        # self.pop_fit is a numpy array
        
        fit_ind_gen = np.argmax(self.pop_fit)
        if self.pop_fit[fit_ind_gen] > self.max_fit:
            self.max_fit = self.pop_fit[fit_ind_gen]
            self.max_fit_img = self.pop[fit_ind_gen]
            self.max_fit_label = str(np.argmax(self.loaded_model.predict(np.array([self.max_fit_img]))))
            if self.max_img:
                print('New best max. fitness: ' + str(self.max_fit))
            if show_image:
                print('Target Image')
                self.show_image(self.target_img)
                print('Population Best Fit')
                print('Classification : ' + self.max_fit_label)
                self.show_image(self.max_fit_img)
        else:
            if self.max_img:
                print('No fitness improvement.')
        
        '''
        for i in range(len(self.pop_fit)):
            if self.pop_fit[i] > max_fit_gen:
                max_fit_gen = self.pop_fit[i]
                max_fit_img = self.pop[i]
                fit_ind_gen = i
        if fit_ind_gen != -1:
            if max_fit_gen > self.max_fit:
                self.max_fit = max_fit_gen;
                self.max_fit_img = max_fit_img
                print('New best max. fitness: ' + str(self.max_fit))
                if show_image:
                    print('Target Image')
                    self.show_image(self.target_img)
                    print('Population Best Fit')
                    print('Classification : ' + str(np.argmax(self.loaded_model.predict(np.array([self.max_fit_img])))))
                    self.show_image(self.max_fit_img)
            else:
                print('No fitness improvement.')
        '''
            
        return
    
    def show_image(self,img):
        #plt.imshow(img.reshape(28,28), cmap='gray')
        #plt.show()
        return
    
    def pick_parents(self):
        #rn = [random.random(), random.random()]
        rn = np.random.rand(2,) #.astype(self.fit_parent_dtype)
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
    
    def pick_parents_pop(self):
        num_parents = 2 * self.pop_cum_prob.shape[0]
        rn = np.random.rand(num_parents)
        rn_argsort = np.argsort(rn)
        parents_sorted = []
        parents = np.empty((2 * self.pop_cum_prob.shape[0],), dtype=np.int64)
        i = 0
        ind = 0
        while len(parents_sorted) < num_parents:
            while rn[rn_argsort[ind]] > self.pop_cum_prob[i]:
                i += 1
            parents_sorted.append(i)
            ind += 1
            while (ind < num_parents) and (rn[rn_argsort[ind]] < self.pop_cum_prob[i]):
                parents_sorted.append(i)
                ind += 1
        parents[rn_argsort] = parents_sorted
        return parents.reshape(self.pop_cum_prob.shape[0], 2)

    
    def compute_c_o_prob(self):
        if self.select_type == 'proportionate':
            self.pop_cum_prob = self.pop_fit / np.sum(self.pop_fit)
            self.pop_cum_prob = np.cumsum(self.pop_cum_prob)
            self.pop_cum_prob[-1] = 1.0
            return
        elif self.select_type == 'rank-linear':
            ranks = np.empty((self.pop_size,))
            ranks[np.argsort(self.pop_fit)] = np.arange(self.pop_size)
            self.pop_cum_prob = ranks / np.sum(ranks)
            self.pop_cum_prob = np.cumsum(self.pop_cum_prob) #.astype(self.fit_parent_dtype)
            self.pop_cum_prob[-1] = 1.0
            return
        elif self.select_type == 'rank-nonlinear':
            ranks = np.empty((self.pop_size,))
            ranks[np.argsort(self.pop_fit)] = np.arange(self.pop_size)
            ranks = self.rank_nonlin_factor**(self.pop_size - ranks)
            self.pop_cum_prob = ranks / np.sum(ranks)
            self.pop_cum_prob = np.cumsum(self.pop_cum_prob) #.astype(self.fit_parent_dtype)
            self.pop_cum_prob[-1] = 1.0
            return
        else:
            print('Invalid select_type')
        
        
    
    '''
    def next_gen(self):
        self.compute_c_o_prob()
        self.crossover()
        self.mutate()
        if self.fit_type == 'mad':
            self.pop_fit = [self.fitness_mad(p) for p in self.pop]
        else:
            self.pop_fit = [self.fitness(p) for p in self.pop]
        self.get_max_fit(self.max_img)
        return
    '''
    
    ''' Function no longer used 
    def mutate_one(self,img):
        if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        img[j] = random.random()  #self.rand_pixel() '''
        
    def mutate_one_pop(self,pop):
        for i in range(len(pop)):
            if random.random() <= self.prob_mut_genome:
                for j in range(len(pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        pop[i][j] = random.random() #self.rand_pixel()
    
    def mutate_one_pop_fast(self, pop):
        phenos = np.where(np.random.rand(len(pop)) <= self.prob_mut_genome)[0]
        for i in phenos:
            pixels = np.where(np.random.rand(len(pop[0]))<= self.prob_mut_pixel)[0]
            for j in pixels:
                pop[i][j] = random.random()

    ''' Function no longer used 
    def mutate_one_mad(self,img):
        if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        img[j] = self.rand_mad_pixel(j) '''
        
    def mutate_one_mad_pop(self,pop):
        for i in range(len(pop)):
            if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        pop[i][j] = self.rand_mad_pixel(j)
    
    def mutate_one_mad_pop_fast(self, pop):
        phenos = np.where(np.random.rand(len(pop))<= self.prob_mut_genome)[0]   
        for i in phenos:
            pixels = np.where(np.random.rand(len(pop[0]))<= self.prob_mut_pixel)[0]
            for j in pixels:
                pop[i][j] = self.rand_mad_pixel(j)    
    
    def next_gen_w_contraint(self):
        ''' Compute crossover probability distribution '''
        #start = time.time()
        self.compute_c_o_prob()
        #self.timer_evolve[0].append(float(time.time() - start))
        
        ''' get best of previous generation '''
        #start = time.time()
        best_dim = min(15,self.pop_size)
        self.sort_ind = np.argsort(-self.pop_fit)[:best_dim].astype(np.int32)
        self.digits = np.argmax(self.loaded_model.predict(self.pop[self.sort_ind]), axis=1)
        
        #digits = [(np.argmax(self.loaded_model.predict(np.array([self.pop[pop_fit_here[i][1]].reshape(784,)]))),i) for i in range(best_dim)]
        self.best_ind = ['' for i in range(10)]
        for i in range(len(self.digits)-1,-1,-1):
            #best_ind[digits[i]] = pop_fit_here[i][1]
            self.best_ind[self.digits[i]] = self.sort_ind[i]
            
        count_good = sum([1 for x in self.best_ind if not isinstance(x, str)])
        #self.timer_evolve[1].append(float(time.time() - start))
        
        #if isinstance(self.best_ind[self.target_label_dig], int):
        #    print('Gen. ' + str(self.num_gen) + '  pop member with label ' + str(self.target_label_dig) + '   count_good: ' + str(count_good))
        
        ''' Selection, crossover and mating '''
        #start = time.time()
        ''' Original list comprehension selection of parents '''
        #parents = [tuple(self.pick_parents()) for i in range(self.pop_size - count_good)]
        ''' Revised selection of parents for entire population in ne function call '''
        parents = self.pick_parents_pop()
        #self.timer_evolve[0].append(float(time.time() - start))
        #start = time.time()
        crossover = np.random.randint(0,self.pop[0].shape[0] - 1, size = (self.pop_size - count_good,))
        #self.timer_evolve[1].append(float(time.time() - start))
        #start = time.time()
        new_pop = [np.concatenate([self.pop[parents[i][0]][0:crossover[i]], self.pop[parents[i][1]][crossover[i]:]]) for i in range(self.pop_size - count_good)]
        #self.timer_evolve[2].append(float(time.time() - start))
        
        ''' Mutation '''
        #start = time.time()
        if self.rand_mode == 'rand':
            #self.mutate_one_pop(new_pop)
            self.mutate_one_pop_fast(new_pop)
        elif self.rand_mode == 'mad':
           #self.mutate_one_mad_pop(new_pop)
           self.mutate_one_mad_pop_fast(new_pop)
        #self.timer_evolve[3].append(float(time.time() - start))
        
        ''' Elitist selection '''
        #start = time.time()
        for i in range(len(self.best_ind)):
            if not isinstance(self.best_ind[i], str):
                new_pop.append(self.pop[self.best_ind[i]])
        #self.timer_evolve[4].append(float(time.time() - start))
        
        ''' Renew population '''
        #start = time.time()
        self.pop_fit_old = self.pop_fit.copy()
        new_pop = np.array(new_pop)    # , dtype = 'float64'    
        #self.timer_evolve[5].append(float(time.time() - start))
        
        ''' Replace infeasible population members with one of their parents '''
        #start = time.time()
        target_match = self.check_target_class_pop(new_pop)
        self.elite_parents_count = 0
        for i in range(len(new_pop)):
            if target_match[i]:
                self.elite_parents_count += 1
                if i > self.pop_size - count_good - 1:
                    print('target_match True for index ' + str(i))
                if random.random() <=0.5:
                    new_pop[i] = self.pop[parents[i][0]]
                else:
                    new_pop[i] = self.pop[parents[i][1]]
        #self.timer_evolve[6].append(float(time.time() - start))
        
        ''' Old code '''
        '''
        while len(new_pop) < self.pop_size - count_good:
            p1,p2 = self.pick_parents()
            cross_pt = random.randint(0,len(self.pop) - 1)   # Possible error: should be self.pop[0]
            # crossover 
            candidate = np.concatenate([self.pop[p1][0:cross_pt], self.pop[p2][cross_pt:]])
            if self.rand_mode == 'rand':
                self.mutate_one(candidate)
            elif self.rand_mode == 'mad':
                self.mutate_one_mad(candidate)
            if self.check_target_class(candidate):
                new_pop.append(self.pop[p1])
                new_pop.append(self.pop[p2])
            else:
                new_pop.append(candidate)'''
        
        #new_pop = new_pop[:self.pop_size - 1]
        #new_pop.append(self.max_fit_img)  #put best fit image from this generation in next generation
                
        self.pop = new_pop.copy()
        #start = time.time()
        self.fitness()
        #self.timer_evolve[7].append(float(time.time() - start))
        
        '''
        if self.fit_type == 'mad-linear':
            self.mad_max = self.get_mad_max()
            self.pop_fit = self.fit_lin_mad() #self.fit_lin_mad()
        elif self.fit_type == 'mad-recip':
            self.pop_fit = self.fit_mad()
        else:
            self.pop_fit = self.fit_ssd() #[self.fitness(p) for p in self.pop]
        
        '''
        #start = time.time()
        self.get_max_fit(self.max_img)
        #self.timer_evolve[3].append(float(time.time() - start))
        
        return 
    
    
    def evolve(self):
        #timer_labels = ['crossover prob','best','selection','mutation','elitist','renew','replace','fitness']
        #self.timer_evolve = [[],[],[],[],[],[],[],[]]
        #timer_labels = ['select parents','crossover','new_pop','get fittest']
        #self.timer_evolve = [[],[],[],[]]
        fs0 = ''
        fs1 = ''
        fs2 = ''
        start_time = datetime.datetime.now()
        for i in range(self.num_gen):
            if self.max_img:
                print('Generation ' + str(i+1) + ':  ', end='')
            self.next_gen_w_contraint()
            
            ''' Print GA statistics to be captured by queue and written to file '''
            ''' Leading 1 indicates GA stats data '''
            ''' 1, MNIST index, generation, pop fitness max, min, average, median '''
            #print(' '.join([str(1), str(self.mnist_index), str(i), str(self.pop_fit_old.max()), 
            #                str(self.pop_fit_old.min()), str(self.pop_fit_old.mean()), str(np.median(self.pop_fit_old))]))
            fs0 += ' '.join([str(self.mnist_index), str(i), str(self.pop_fit_old.max()), str(self.pop_fit_old.min()), 
                             str(self.pop_fit_old.mean()), str(np.median(self.pop_fit_old))]) + '\n'
            ''' Leading 2 indicates elite_parents stats data '''
            ''' 2, MNIST index, generation, pop fitness max, min, average, median '''
            #print(' '.join([str(2), str(self.mnist_index), str(i), str(self.elite_parents_count)]))
            fs1 += ' '.join([str(self.mnist_index), str(i), str(self.elite_parents_count)]) + '\n'
            ''' Leading 3 indicates elite stats data '''
            ''' 3, MNIST index, generation, pop fitness max, min, average, median '''
            for j in range(len(self.best_ind)):
                if not isinstance(self.best_ind[j], str):
                    #print(' '.join([str(3), str(self.mnist_index), str(i), str(j), str(self.pop_fit_old[self.best_ind[j]])]))
                    fs2 += ' '.join([str(self.mnist_index), str(i), str(j), str(self.pop_fit_old[self.best_ind[j]])]) + '\n'
            
            ''' Write Results to File '''
            #f_out.write(str(i) + ',' + str(self.max_fit) + ',' + str(np.mean(self.pop_fit)) + '\n')
            
        ''' Write best-fit image to numpy file '''
        with open(self.out_folder + str(self.mnist_index) + '_img' + '.npy', 'wb') as f:
            np.save(f, self.max_fit_img)
        
        ''' Write GA stats to file '''
        with open(self.out_folder + str(self.mnist_index) + '_pop_stat' + '.csv', 'w') as f:
            f.write(fs0)
        with open(self.out_folder + str(self.mnist_index) + '_elite_parents' + '.csv', 'w') as f:
            f.write(fs1)
        with open(self.out_folder + str(self.mnist_index) + '_elite' + '.csv', 'w') as f:
            f.write(fs2)
            
        #timer_out = ''
        #for i in range(len(self.timer_evolve)):
        #    timer_out += timer_labels[i] + ': ' + str(sum(self.timer_evolve[i]))  + ' seconds\n'
        #with open(self.out_folder + str(self.mnist_index) + '_timer_detail' + '.csv', 'w') as f:
        #    f.write(timer_out)
        
        ''' Write Best Fit Results to File'''
        ''' These are no longer output to file '''
        '''
        combine_fit_pop = [(self.pop_fit[i], self.pop[i], np.argmax(self.loaded_model.predict(np.array([self.pop[i]])))) for i in range(self.pop_size)]
        combine_fit_pop.sort(reverse=True, key = lambda x:x[0])
        
        done = False
        max_fit = combine_fit_pop[0][0]
        digit_done = [False for i in range(10)]
        best_images = np.array([combine_fit_pop[0][1]])
        digit_done[combine_fit_pop[0][2]] = True
        i = 1
        best_labels = [combine_fit_pop[0][2]]
        while not done:
            if combine_fit_pop[i][0] > 0.90 * max_fit and i < len(combine_fit_pop) - 1:
                if digit_done[combine_fit_pop[i][2]] == False:
                    digit_done[combine_fit_pop[i][2]] = True
                    best_images = np.vstack([best_images,combine_fit_pop[i][1]])
                    best_labels.append(combine_fit_pop[i][2])
                i += 1
            else:
                done = True
        '''
        '''
        f = open('./output/images/' + self.filename + '_fit.csv','w')
        for i in range(1,15):
            f.write(str(combine_fit_pop[i][0]) + '\n')
            best_images = np.vstack([best_images,combine_fit_pop[i][1]])
        f.close()
        np.savetxt('./output/images/' + self.filename + '_images.csv', best_images, delimiter=",",fmt='%8.6f',newline='\n')
        '''
        
        ''' Compute timing '''
        finish_time = datetime.datetime.now()
        elapse_time = finish_time - start_time
        
        ''' Report Results to console '''
        if self.max_img:
            print('\n\n\n\nResults')
            print('Maximum fitness: ' + str(self.max_fit) + '\n')
            print('Target Image')
            self.show_image(self.target_img)
                
            '''
            print('Best Fit Images')
            for i in range(len(best_images)):
                print('Image label:' + str(best_labels[i]))
                self.show_image(best_images[i])
            '''
        
        
        ''' Put best into database '''
        '''
        cnx = MySQL.connect(user='root', passwd='MySQL', host='127.0.0.1', db='adv_exmpl')
        cursor = cnx.cursor(buffered=True)
        for i in range(len(best_images)):
            img_str = ''
            for j in range(best_images[i].shape[0]):
                img_str += str(best_images[i][j]) + ' '
            cursor.callproc('spInsertRow',(self.mnist_index,int(self.target_label_dig),int(best_labels[i]), img_str)) # best_images[i].tobytes()
            cnx.commit() '''
            
        
        
        
        ''' Create output -- old output with all best image '''
        '''
        output = []
        for i in range(len(best_images)):
            img_str = ''
            for j in range(best_images[i].shape[0]):
                img_str += str(best_images[i][j]) + ' '
            output.append(str(self.mnist_index) + ', ' + str(self.target_label_dig) + ', ' + str(best_labels[i]) + ', ' + str(self.max_fit) + ',' + str(elapse_time) + ', ' + img_str + '\n')'''
        
        ''' New output with only best image '''
        img_str = ''
        for j in range(self.max_fit_img.shape[0]):
            img_str += str(self.max_fit_img[j]) + ' '
        output = str(self.mnist_index) + ', ' + str(self.target_label_dig) + ', ' + str(self.max_fit_label) + ', ' + str(self.mad_max) + ', ' + str(self.max_fit) + ',' + str(elapse_time) + ', ' + img_str
        
        ''' Write to log file '''
        #finish_time = datetime.datetime.now()
        #elapse_time = finish_time - start_time
        #minutes = int(elapse_time.total_seconds()/60)
        #seconds = elapse_time.total_seconds() - minutes * 60.0
        #f_out = open(self.out_folder + self.log_name + '.log','a')
        #f_out.write('MNIST index ' + str(self.mnist_index) + ' completed at ' + datetime.datetime.strftime(datetime.datetime.now(), '%m/%d/%y %H:%M:%S') + ' in ' + str(minutes) + ':' + str(seconds) + ' with fitness ' + str(self.max_fit) + '\n')
        #f_out.close()
        
        #new_log_name = re.sub('E\d+', 'E' + str(self.mnist_index), self.log_name)
        
        #os.rename(self.out_folder + self.log_name + '.csv', self.out_folder + new_log_name + '.csv')
        #os.rename(self.out_folder + self.log_name + '.log', self.out_folder + new_log_name + '.log')
        
        #self.log_name = new_log_name
        
        return output
        
    ''' Function no longer used
    def mutate(self):
        for i in range(len(self.pop)):
            if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        self.pop[i,j] = self.rand_pixel()
        return '''
    
    ''' crossover() function no longer used 
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
        return '''

    def check_target_class(self, img):
        return np.argmax(self.loaded_model.predict(np.array([img]), batch_size=1000)) == self.target_label_dig
    
    def check_target_class_pop(self, pop):
        return np.argmax(self.loaded_model.predict(pop, batch_size=1000), axis = 1) == self.target_label_dig
    
""" Function Definitions """

""" Create random grayscale images with pixels on [0.0, 1.0] """
''' Function no longer used '''
'''
def randDig():
    blkProb = 0.7
    whtProb = 0.25/(1 - blkProb)
    #img =  [[ 0.0 if random.random() <= blkProb else 1.0 if random.random() <= whtProb else random.randint(1,255)/255 for j in range(28)] for i in range(28)]
    img =  [0.0 if random.random() <= blkProb else 1.0 if random.random() <= whtProb else random.randint(1,255)/255 for i in range(784)]
    return np.array(img)'''

'''
def fitness(bench,cand):
    ssq = 0.0
    for i in range(len(bench)):
        ssq += (bench[i] - cand[i])^2
    return 1/ssq '''
    
''' function no longer used '''
'''
def fitness(target,cand):
    return 1/np.sum((target - cand)**2) '''



''' Handle input arguments '''
parser = argparse.ArgumentParser(description='Generate adversarial examples for neural network')
parser.add_argument('mnist_id', metavar='mnist_id', type=int, help='MNIST index for evaluation')
parser.add_argument('fit_type', metavar='fit_type', type=str, help='Fitness function type')
parser.add_argument('select_type', metavar='select_type', type=str, help='Selection method for parents')
parser.add_argument('rand_type', metavar='select_type', type=str, help='Randomization mode for mutation and initializing populations')
parser.add_argument('factor_rank_nonlinear', metavar='factor_rank_nonlinear', type=float, help='Factor for nonlinear rank selectio')
parser.add_argument('file_model', metavar='file_model', type=str, help='JSON file for neural network model')
parser.add_argument('file_weights', metavar='file_weights', type=str, help='h5 file for neural network weights')
parser.add_argument('out_folder', metavar='out_folder', type=str, help='file path for output')
parser.add_argument('in_folder', metavar='folder', type=str, help='file path for input')
parser.add_argument('pop_size', metavar='folder', type=int, help='Population size')
parser.add_argument('prob_mut_genome', metavar='folder', type=float, help='Probability of phenotype mutation')
parser.add_argument('prob_mut_pixel', metavar='folder', type=float, help='Probability of pixel mutation')
parser.add_argument('num_gen', metavar='folder', type=int, help='Number of generations')
parser.add_argument('scen_name', metavar='folder', type=str, help='Scenario name for output file folder')
args = parser.parse_args()

''' Set GA parameters '''
pop_size = args.pop_size  #1000   # population size
prob_mut_genome = args.prob_mut_genome  #0.7    # probability of phenotype mutation
#genome_size = 784
prob_mut_pixel = args.prob_mut_pixel  #2.0 / genome_size  # probability of pixel mutation
num_gen = args.num_gen #2000     # number of generations
max_fit = 0        # maximum fitness
#pop_fit = []      # population fitness
#target_index = 10
#model_filename = args.file_model #'ff_mnist.json'
#weights_filename = args.file_weights #'ff_mnist.h5'
prob_wht = 0.25
prob_blk = 0.7
min_mad = 0.1    # Values tried: 0.001, 0.05
#fit_type = 'mad-linear'  # 'mad-recip' 'L1'  'L2'  'Linf'
#rand_type = 'mad' 'rand'
#select_type = 'rank-nonlinear'  # 'rank-linear' 'proportionate' 'rank-nonlinear'
        

#input_folder = args.folder +  'input/' #'/sciclone/home10/jrbrad/files/mnist/input/'
#input_folder = 'C:/Users/jrbrad/Desktop/adversarial_eg/ga/input/'
#output_folder = args.out_folder #'/sciclone/home10/jrbrad/files/mnist/output/'
prints_img = False
    
#log_name = re.sub('_','-',re.sub('\.json','',args.file_model)) + '_' + re.sub('_','-',re.sub('\.h5','',args.file_weights)) + '_' + 'pop' + str(pop_size) + '_' + 'mutgenome' + str(int(prob_mut_genome*100)) + '_' + 'mutpix' + str(int(prob_mut_pixel*1000)) + '_' + 'gen' + str(num_gen) + '_' + 'fit-' + fit_type + '_' + 'rand-' + rand_type + '_B' + str(args.start) + 'E' + str(args.start)
''' Create empty output file '''
#f_out = open(output_folder + log_name + '.csv','w')
#f_out.close()
''' Create empty log file '''
#f_out = open(output_folder + log_name + '.log','w')
#f_out.close()

#scen_name = re.sub('_','-',re.sub('\.json','',args.file_model)) + '_' + str(pop_size) + '_' + str(num_gen)  + '_' + str(int(prob_mut_genome*100)) + '_'+ str(int(prob_mut_pixel*1000))  + '_' + args.fit_type  + '_' + args.select_type + '_' + args.rand_type

''' Instantiate GA object '''
#ga = GA(pop_size, num_gen, prob_mut_genome, prob_mut_pixel, prob_wht, prob_blk, num_gen, i, input_folder, output_folder, loaded_model, prints_img, log_name, fit_type, min_mad, 'mad')
ga = GA(pop_size, num_gen, prob_mut_genome, prob_mut_pixel, prob_wht, prob_blk, num_gen, args.in_folder, args.file_model, args.file_weights, args.out_folder, prints_img, args.fit_type, args.select_type, min_mad, args.rand_type, args.factor_rank_nonlinear, args.scen_name)
ga.new(args.mnist_id)
result = ga.evolve()
'''
for i in range(len(result)):
    result[i] = scen_name + ',' + result[i]'''
print(args.scen_name + ',' + result, end='')