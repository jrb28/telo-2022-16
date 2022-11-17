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


def run_this(q, mnist_id, fit_type, select_type, rand_type, fit_nonlin_rank, model_file, weights_file, 
             out_folder, in_folder, filepath_code, pop_size, prob_mut_genome, prob_mut_pixel, num_gen, 
             scen_name):
    result = subprocess.run(['python', filepath_code , str(mnist_id), fit_type, select_type, rand_type, str(fit_nonlin_rank), model_file, weights_file, out_folder, in_folder, str(pop_size), str(prob_mut_genome), str(prob_mut_pixel), str(num_gen), scen_name], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    q.put(result)
    return result

def run_this_no_q(mnist_id, fit_type, select_type, rand_type, fit_nonlin_rank, model_file, weights_file, 
                  out_folder, in_folder, filepath_code, pop_size, prob_mut_genome, prob_mut_pixel, num_gen, 
                  scen_name):
    result = subprocess.run(['python', filepath_code , str(mnist_id), fit_type, select_type, rand_type, str(fit_nonlin_rank), model_file, weights_file, out_folder, in_folder, str(pop_size), str(prob_mut_genome), str(prob_mut_pixel), str(num_gen), scen_name], stdout=subprocess.PIPE)
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
    parser.add_argument('rand_type', metavar='select_type', type=str, help='Randomization mode for mutation and initializing populations: rand = uniformly random pixel values on [0.0,1.1]; mad = drawn from observed distribution of pixel values')
    parser.add_argument('factor_rank_nonlinear', metavar='factor_rank_nonlinear', type=float, help='Factor for nonlinear rank selectio')
    parser.add_argument('mp_mode', metavar='mp_mode', type=str, help='Multiprocessing mode (Boolean)')
    parser.add_argument('num_proc', metavar='num_proc', type=str, help='Number of processes (integer)')
    parser.add_argument('file_model', metavar='file_model', type=str, help='JSON file for neural network model')
    parser.add_argument('file_weights', metavar='file_weights', type=str, help='h5 file for neural network weights')
    parser.add_argument('folder_in', metavar='folder', type=str, help='File path for input')
    parser.add_argument('folder_out', metavar='folder', type=str, help='File path for output')
    parser.add_argument('filepath_code', metavar='folder', type=str, help='File path for code')
    parser.add_argument('pop_size', metavar='folder', type=int, help='Population size')
    parser.add_argument('prob_mut_genome', metavar='folder', type=float, help='Probability of phenotype mutation')
    parser.add_argument('pixel_mut_per_phenotype', metavar='folder', type=float, help='Probability of phenotype mutation')
    parser.add_argument('num_gen', metavar='folder', type=int, help='Number of generations')
    args = parser.parse_args()
    mp_mode = str_to_bool(args.mp_mode)
    start = time.time()
    
    genome_size = 784
    prob_mut_pixel =  args.pixel_mut_per_phenotype / genome_size  # 2.0 / genome_size  # probability of pixel mutation
    scen_name = re.sub('_','-',re.sub('\.json','',args.file_model)) + '_' + str(args.pop_size) + '_' + str(args.num_gen)  + '_' + str(int(args.prob_mut_genome*100)) + '_'+ str(int(prob_mut_pixel*1000))  + '_' + args.fit_type  + '_' + args.select_type + '_' + args.rand_type
    
    ''' Create folder for scenario (scen_name) output files if it doesn't exist '''
    if not os.path.isdir(args.folder_out + scen_name):
        os.mkdir(args.folder_out + scen_name)
    
    ''' Added statements for picking up missed mnist images '''
    in_type = 'range'
    if in_type == 'json':
        underscore_strip = re.compile('_\w*')
        file_name_missing = underscore_strip.sub('', args.file_model.rstrip('.json'))
        #f = open(args.folder+'input/'+file_name_missing+'_miss.csv')
        f = open(args.folder_in+file_name_missing+'_miss.csv')
        indices = f.readlines()
        f.close()
        for i in range(len(indices)):
            indices[i] = int(indices[i])
    elif in_type == 'range':
        indices = list(range(args.start_id, args.end_id+1))
    num_progs = len(indices)
    
    filename_stub = re.sub('.json','',args.file_model) + '_' + args.fit_type + '_' + args.select_type  + '_' + args.rand_type  + '_' 
        
    nums = re.compile('[0-9]+\.csv')
    files = glob.glob(args.folder_out + filename_stub + '*.csv')
    if len(files) == 0:
      ext = str(0)
    else:
        for i in range(len(files)):
            try:
                files[i] = int(nums.search(files[i]).group(0).rstrip('.csv'))
            except:
                pass
        files = [x for x in files if isinstance(x, int)]
        ext = str(max(files) + 1)
    
    
    
    output_file = args.folder_out + filename_stub + ext + '.csv'
    f = open(output_file,'w')
    f.write('')
    f.close()
    
    if mp_mode:
        try:
            num_proc = int(args.num_proc) #18 #4
            print('Creating pool with %d processes\n' % num_proc)
        except:
            print('Argument for number of processes cannot be converted to an integer. \n')
        
        with multiprocessing.Pool(num_proc) as pool:
        
            m = multiprocessing.Manager()
            q = m.Queue()
            
            TASKS = [(q, i, args.fit_type, args.select_type, args.rand_type,  args.factor_rank_nonlinear, args.file_model, args.file_weights, args.folder_out, args.folder_in, args.filepath_code, args.pop_size, args.prob_mut_genome, prob_mut_pixel, args.num_gen, scen_name) for i in indices]  #range(args.start_id, args.end_id + 1)
            
                
            results1 = pool.starmap_async(run_this, TASKS)
            print('DONE')
            
            num_retrieve = 0
            while num_retrieve < num_progs:
                try:
                    result = q.get()
                    #if result != None:
                    print("result", num_retrieve, ":", result)
                    num_retrieve += 1
                    #result = result.rstrip(']').lstrip('[').split(',')
                    f = open(output_file,'a')  #, buffering=0
                    f.write(result)
                    f.write('\n')
                    f.close()
                except:
                    time.sleep(1)
                    
        end = time.time()
        with open(args.folder_out + scen_name + '_timing.txt','w') as f:
            f.write(str(float(end) - float(start)))
    
                    
    else:
        results1 = []
        for t in [(i, args.fit_type, args.select_type, args.rand_type,  args.factor_rank_nonlinear, args.file_model, args.file_weights, args.folder_out, args.folder_in, args.filepath_code, args.pop_size, args.prob_mut_genome, prob_mut_pixel, args.num_gen, scen_name) for i in indices]:
            results1.append(run_this_no_q(*t))
            
        '''Write results to file '''
        results1 = [r.lstrip('[').rstrip(']').strip("'").strip().replace('\\n','\n') for r in results1]
        print(results1)
        
        '''Write results to file '''
        with open(output_file,'w') as f:
            f.write(''.join(results1))
        
        params = args.fit_type + '_' + args.select_type + '_' + args.rand_type + '_' + str(args.factor_rank_nonlinear) + '_'
        with open(args.folder_out + params + '_' + str(args.start_id) + '_' + str(args.end_id) + '.csv', 'w') as f:
            f.write(f'Execution time: {float(time.time() - start)/60} minutes for {args.end_id - args.start_id + 1} images')
        
    
    print('Done.')