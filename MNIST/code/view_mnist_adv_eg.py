# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 08:32:21 2022

@author: james
"""

import matplotlib.pyplot as plt
import numpy as np

img_path = '../output/y/1_img.npy'

img = np.load(img_path).reshape(28,28)

fig,ax = plt.subplots()
ax.imshow(img, cmap='gray')
plt.show()