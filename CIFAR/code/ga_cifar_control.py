# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:31:22 2020

@author: 
"""

'''
Copyright 2022 by Anonymous Authors
This code is protected by the Academic Free license Version 3.0
See LICENSE.md in the repository root folder for the terms of this license
'''

import subprocess
import multiprocessing
import time
import argparse
import glob
import re
import os
import json


def run_this(q, mnist_id, model_file, out_folder, folder, fit_type, select_type, rand_type, factor_rank_nonlinear, gpu_mode):
    #path = pathlib.Path(__file__).parent.absolute()
    result = subprocess.run(['python', 'ga_cifar_worker.py', str(mnist_id), model_file, out_folder, folder, 
                             fit_type, select_type, rand_type, str(factor_rank_nonlinear), gpu_mode], 
                            stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    q.put(result)
    return result

def run_this_no_q(mnist_id, model_file, out_folder, folder, fit_type, select_type, rand_type, factor_rank_nonlinear, gpu_mode):
    result = subprocess.run(['python', 'ga_cifar_worker.py', str(mnist_id), model_file, out_folder, folder, 
                             fit_type, select_type, rand_type, str(factor_rank_nonlinear), gpu_mode], 
                            stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    return result

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate adversarial examples for neural network')
    parser.add_argument('start_id', metavar='start_id', type=int, help='Starting MNIST index for evaluation')
    parser.add_argument('end_id', metavar='end_id', type=int, help='Ending MNIST index for evaluation')
    parser.add_argument('fit_type', metavar='fit_type', type=str, help='Fitness function type')
    parser.add_argument('select_type', metavar='select_type', type=str, help='Selection method for parents')
    parser.add_argument('rand_type', metavar='rand_type', type=str, help='Randomization mode for mutation and initializing populations: rand = uniformly random pixel values on [0.0,1.1]; mad = drawn from observed distribution of pixel values')
    parser.add_argument('factor_rank_nonlinear', metavar='factor_rank_nonlinear', type=float, help='Factor for nonlinear rank selectio')
    parser.add_argument('num_proc', metavar='num_proc', type=str, help='Number of processes')
    parser.add_argument('file_model', metavar='file_model', type=str, help='JSON file for neural network model')
    parser.add_argument('folder', metavar='folder', type=str, help='base file folder for input')
    parser.add_argument('folder_out', metavar='folder', type=str, help='file folder for output')
    parser.add_argument('gpu_mode', metavar='gpu_mode', type=str, help='GPU/CPU prediction mode (Boolean)')
    parser.add_argument('mp_mode', metavar='mp_mode', type=str, help='Multiprocessing mode (Boolean)')
    parser.add_argument('batch_id', metavar='batch_id', type=str, help='Batch ID')
    args = parser.parse_args()
    
    time_start = time.time()
    
    rand_indices = False
    mp_mode = str_to_bool(args.mp_mode)
    
    ''' Select consecutive or random indices '''
    if rand_indices:
        with open(args.folder + 'data/randCIFAR.json', 'r') as f:
            indices = json.load(f)
            indices = indices[args.start_id : args.end_id + 1]
            num_progs = len(indices)
    else:
        num_progs = args.end_id - args.start_id + 1
        indices = range(args.start_id , args.end_id + 1)
    
    
    model_file = args.file_model    #'ff_mnist.json'
    filename_stub = re.sub('.json','',model_file) 
        
    
    ''' set up output file folder and file name '''
    output_file = args.folder_out + args.batch_id + '.csv'
    
    if mp_mode:
        try:
            num_proc = int(args.num_proc) #18 #4
            print('Creating pool with %d processes\n' % num_proc)
        except:
            print('Argument for number of processes cannot be converted to an integer. \n')
    
        with multiprocessing.Pool(num_proc) as pool:
        
            m = multiprocessing.Manager()
            q = m.Queue()
            
            TASKS = [(q, i, model_file, args.folder_out, args.folder, args.fit_type, args.select_type, args.rand_type, args.factor_rank_nonlinear, args.gpu_mode) for i in indices]
    
            results1 = pool.starmap_async(run_this, TASKS)
            print('Done assigning to pool')
            pool.close()
            pool.join()
            print('Done computing')
                    
            ''' Convert starmap results to list of strings '''
            results1 = [r for r in results1.get()]
    else:
        results1 = []
        for t in [(i, model_file, args.folder_out, args.folder, args.fit_type, args.select_type, args.rand_type, args.factor_rank_nonlinear, args.gpu_mode) for i in indices]:
            results1.append(run_this_no_q(*t))
    
    '''Write results to file '''
    results1 = [r.lstrip('[').rstrip(']').strip("'").strip().replace('\\n','\n') for r in results1]
    print(results1)
    
    '''Write results to file '''
    with open(output_file,'w') as f:
        f.write(''.join(results1))
    
    params = args.fit_type + '_' + args.select_type + '_' + args.rand_type + '_' + str(args.factor_rank_nonlinear) + '_'
    with open(args.folder_out + params + '_' + str(args.start_id) + '_' + str(args.end_id) + '.csv', 'w') as f:
        f.write(f'Execution time: {float(time.time() - time_start)/60} minutes for {args.end_id - args.start_id + 1} images')
    
    print('Done.')