# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:13:27 2021

@author: james
"""

import argparse
import json
import subprocess

'''
def fgsm(num_mnist_labels, mnist_id, input_folder, output_folder, model_filename, weights_filename, 
         fgsm_worker_filename, epsilon, epsilon_delta, show_images):
    # q, 
    result = subprocess.run(['python', fgsm_worker_filename , str(num_mnist_labels), str(mnist_id), 
                             input_folder, output_folder, model_filename, weights_filename, str(epsilon), str(epsilon_delta), 
                             str(show_images)], stdout=subprocess.PIPE)
    #result = subprocess.run(['python', 'C:/Users/jrbrad/Desktop/adversarial_eg/ga/cl/new/ga_mnist_adv_worker.py', str(mnist_id), model_file, weights_file, out_folder], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    print(result)
    #q.put(result)
    return result

def check(*args):
    print('x')
    return 'x'
'''

if __name__ == '__main__':
    ''' Sample arguments: ./input/ ./output/ randMNIST.json FF.json FF.h5 fgsm_wkr_simple.py '''
    parser = argparse.ArgumentParser(description='Create Fast Gradient Sign Method (FGSM) adversarial examples.')
    parser.add_argument('input_folder', metavar='input_folder', type=str, help='(Relative) input folder path')
    parser.add_argument('output_folder', metavar='output_folder', type=str, help='(Relative) output folder path')
    parser.add_argument('mnist_rand_indices_filename', metavar='mnist_rand_indices', type=str, help='Filename containing randomly selected MNIST images')
    parser.add_argument('model_filename', metavar='model_filename', type=str, help='Filename for neural network graph')
    parser.add_argument('weights_filename', metavar='weights_filename', type=str, help='Filename for neural network graph weights')
    parser.add_argument('fgsm_worker_filename', metavar='fgsm_worker_filename', type=str, help='Filename for FGSM work')
    args = parser.parse_args()
    
    
    ''' Parameters '''
    num_proc = 1
    num_mnist_labels = 10
    debug = False
    
    ''' Create local variables for arguments '''
    in_folder = args.input_folder
    out_folder = args.output_folder
    mnist_rnd_idx = args.mnist_rand_indices_filename
    model_filename = args.model_filename
    weights_filename = args.weights_filename
    fgsm_worker = args.fgsm_worker_filename
    
    ''' Load randomly selected MNIST image indices '''
    mnist_idx = json.load(open(in_folder + mnist_rnd_idx, 'r'))
    mnist_idx.sort()
    if debug: mnist_idx = mnist_idx[:4]
    
    arg_list = [(str(num_mnist_labels), str(mnist_id), in_folder, out_folder, model_filename, weights_filename,
                 '0.01', '0.01', '0') for mnist_id in mnist_idx]
    
    ''' 10 0 ./input/ ./output/ FF.json FF.h5  0.01 0.01 0 '''
    
    
    results1 = ''
    for arg in arg_list:
        #result = subprocess.run(['python', fgsm_worker , str(num_mnist_labels), str(mnist_id), 
        #                     input_folder, output_folder, model_filename, weights_filename, str(epsilon), str(epsilon_delta), 
        #                     str(show_images)], stdout=subprocess.PIPE)
        result = subprocess.run(['python', fgsm_worker, *arg], stdout=subprocess.PIPE)
        result = result.stdout.decode('utf-8')
        print(result)
        #results1 += fgsm(*arg)
        results1 += result
    
    try:
        with open(out_folder + 'fgsm_results_' + fgsm_worker.replace('.py','').split('_')[-1] + '.csv', 'a') as f:
            f.write(results1.replace('\r\n','\n'))
    except:
        with open(out_folder + 'fgsm_results_' + fgsm_worker.replace('.py','').split('_')[-1] + '.csv', 'w') as f:
            f.write(results1.replace('\r\n','\n'))        