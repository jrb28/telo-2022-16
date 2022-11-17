# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:13:27 2021

@author: james
"""

import keras
from keras import models
from keras.utils import to_categorical
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

loss_f = keras.losses.CategoricalCrossentropy()

def grad_sign(model, input_image, input_label):
    with tf.GradientTape() as t:
        t.watch(input_image)
        prediction = model(input_image)
        loss = loss_f(input_label, prediction)
        dy_dx = tf.gradients(loss, input_image)
        #print(dy_dx, type(dy_dx))
        #print(prediction)
        #with tf.Session() as sess:
        #    init = tf.global_variables_initializer()
        #    sess.run(init)
        #    print(prediction.eval())
    return tf.sign(dy_dx)
    
def tensor2np(tensor):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        x = tensor.eval().reshape(28,28)
    return x

parser = argparse.ArgumentParser(description='Create Fast Gradient Sign Method (FGSM) adversarial example for a given MNIST image.')
parser.add_argument('num_mnist_labels', metavar='num_mnist_labels', type=int, help='Number of distinct MNIST labels')
parser.add_argument('mnist_id', metavar='mnist_id', type=int, help='MNIST image ID')
#parser.add_argument('mnist_img_filename', metavar='mnist_image', type=np.float32, help='MNIST image data (numpy array)')
parser.add_argument('input_folder', metavar='input_folder', type=str, help='(Relative) input folder path')
parser.add_argument('output_folder', metavar='output_folder', type=str, help='(Relative) output folder path')
parser.add_argument('model_filename', metavar='model_filename', type=str, help='Neural network graph filename')
parser.add_argument('weights_filename', metavar='weights_filename', type=str, help='Neural network weights filename')
#parser.add_argument('mnist_label', metavar='mnist_label', type=int, help='MNIST image label')
parser.add_argument('epsilon', metavar='epsilon', type=float, help='Gradient step size')
parser.add_argument('epsilon_delta', metavar='epsilon_delta', type=float, help='Delta gradient step size')
parser.add_argument('show_images', metavar='show_images', type=int, help='Boolean flag: if True, then show images and iteration data')
args = parser.parse_args()

num_mnist_labels = args.num_mnist_labels
mnist_id = args.mnist_id
model_filename = args.model_filename
weights_filename = args.weights_filename
#mnist_label = args.mnist_label
in_folder = args.input_folder
out_folder = args.output_folder
eps = args.epsilon
eps_delta = args.epsilon_delta

start = time.time()

''' Load and preprocess MNIST image and label '''
(train_images, train_labels), (_, _) = mnist.load_data()
img = tf.constant(train_images[mnist_id].reshape(1,784) / 255, dtype = np.float32)
mnist_label = train_labels[mnist_id]
mnist_label_vec = to_categorical(mnist_label, num_classes=num_mnist_labels)
predict_new = mnist_label
del train_images
del train_labels

'''
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))
train_images = train_images / 255  #.astype(self.fit_parent_dtype)
test_images = test_images / 255  # .astype(self.fit_parent_dtype)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)'''

''' Load Model & Weights, and Compile '''
#model_filename = in_folder + model_filename
#weights_filename = in_folder + weights_filename
#in_folder = './input/'
json_file = open(in_folder + model_filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
#model_abbrv = model_filename.replace('.json','')
loaded_model = models.model_from_json(loaded_model_json)
loaded_model.load_weights(in_folder + weights_filename)
loaded_model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

''' The statements immediately below tests the proper way to predict a MNIST label '''
''' The statement below works '''
#predict_all = np.argmax(loaded_model.predict(train_images), axis = 1)
''' doesn't work without the .predict() method perhaps because it is symbolic rather than computational '''
#predict_all_1 = np.argmax(loaded_model(tf.constant(train_images, dtype=np.float32)), axis = 1)

''' compute gradient '''
#mnist_index = 0
#num_mnist_labels = 10
#img = tf.constant(train_images[mnist_index].reshape(1,784), dtype = np.float32)
#img_new = tf.constant(train_images[mnist_index].reshape(1,784), dtype = np.float32)

g = grad_sign(loaded_model, img, mnist_label_vec)
steps = -1
#eps = 0.01
#eps_delta = 0.01
#predict = np.argmax(train_labels[mnist_index])
#predict_new = np.argmax(train_labels[mnist_index])

while (mnist_label == predict_new) and eps  + steps * eps_delta < 1.0:
    #eps += eps_delta
    steps += 1
    img_new = tf.clip_by_value(img + (eps  + steps * eps_delta) * g, 0, 1)[0]
    #predict = np.argmax(loaded_model.predict(img, steps=1)[0])
    predict_new = np.argmax(loaded_model.predict(img_new, steps=1)[0])
    
    
    if args.show_images:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(tensor2np(img), cmap = 'gray')
        ax[1].imshow(tensor2np(img_new), cmap = 'gray')
        plt.show()
        print('Original image prediction:', mnist_label)
        print('Perturbed image prediction:', predict_new)
        print('epsilon: %.2f' % (eps,))
    
np.savetxt(out_folder + 'fgsm_' + model_filename.split('.')[0] + '_'  + __file__.split('\\')[-1].split('.')[0].split('_')[-1] + '_' + str(mnist_id) + '.npy', tensor2np(img_new).reshape((1,784)))    
   
print(f'{mnist_id:d}, {mnist_label:d}, {predict_new:d}, {eps  + steps * eps_delta:.3f}, {time.time() - start:10.7f}, {steps + 1:d}')
#print(f'Execution time: {time.time() - start} seconds.')
keras.backend.clear_session()