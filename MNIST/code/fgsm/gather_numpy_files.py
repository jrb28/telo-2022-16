# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 08:44:16 2021

@author: james
"""

import json
import argparse
import numpy as np
''' Sample arguments: ./input/ ./output/ randMNIST.json fgsm_FF_simple_ '''

parser = argparse.ArgumentParser(description='Create Fast Gradient Sign Method (FGSM) adversarial examples.')
parser.add_argument('input_folder', metavar='input_folder', type=str, help='(Relative) input folder path')
parser.add_argument('output_folder', metavar='output_folder', type=str, help='(Relative) output folder path')
parser.add_argument('mnist_rand_indices_filename', metavar='mnist_rand_indices', type=str, help='Filename containing randomly selected MNIST images')
parser.add_argument('filename_template', metavar='filename_template', type=str, help='Filename containing randomly selected MNIST images')
args = parser.parse_args()    

in_folder = args.input_folder
out_folder = args.output_folder
mnist_rnd_idx = args.mnist_rand_indices_filename
filename_templ = args.filename_template #'fgsm_FF_simple_'

mnist_idx = json.load(open(in_folder + mnist_rnd_idx, 'r'))
mnist_idx.sort()

data = []
for mnist_id in mnist_idx:
    print(mnist_id)
    data.append(np.loadtxt(out_folder + filename_templ + f'{mnist_id:d}.npy').reshape((1,784)))

print(f'{len(data):d} files found')
data = np.concatenate(data)    
#np.savetxt(out_folder + filename_templ[:-1] + '.npy', data)
np.save(out_folder + filename_templ[:-1] + '.npy', data)

''' check count of MNIST IDs in file '''
with open('./input/randMNIST.json', 'r') as f:
    data = json.load(f)
print(f'{len(data):d} MNIST IDs found')