# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:44:06 2021

@author: james
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import  mnist

index = 22
folder = 'FF_1000_2000_70_2_L2_rank-nonlinear_rand'

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(60000, 784).astype(np.float32)/255

fig,ax = plt.subplots(1,2)
ax[0].imshow(train_images[index].reshape(28,28), cmap='gray')
ax[1].imshow(np.load(f'../output/{folder}/{str(index)}_img.npy').reshape(28,28), cmap='gray')
plt.show()