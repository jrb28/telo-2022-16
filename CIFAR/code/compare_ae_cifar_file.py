# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:15:02 2022

@author: james
"""

import matplotlib.pyplot as plt
import numpy as np

''' Load CIFAR-10 images '''
img_cifar = np.load('../adv_egs/cifar_rand_images.npy')
scen = 'L2-lin_rank-nonlinear'
adv_eg = np.load(f'../adv_egs/{scen}.npy')

img_id = 0

fig,ax = plt.subplots(1,2)
ax[0].imshow(img_cifar[img_id].reshape(32,32,3))
ax[1].imshow(adv_eg[img_id].reshape(32,32,3))
plt.show()