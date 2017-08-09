
# coding: utf-8

# In[7]:

import functools
from functools import reduce
from typing import Dict, Callable
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any, Union
#from functional import f_filter, f_map, f_reduce
from scipy import signal
from glia.types import Unit
from tqdm import tqdm
import random    # added to Tyler's
#import analysis
from scipy import ndimage
import math


'''
This toy plotter shows what the plots could look like once the function is complete.

It assumes each cell is at the center of the array, and that recepeptive fields could
cover the entire area of the stimulus image.
Uses random values for response times.
'''

# *********************** generate fake data using artifical units, then plot receptive fields ********************
num_units = 47
num_cols = 6         # choose based on desired size/spread of plots


# ************ randomized fake data *************
theta = np.linspace(0, 2*np.pi, 12)  # generates 12 evenly spaced angles; these are NOT correct
t = np.empty((num_units, 12))
for i in range(num_units):
    for j in range(12):              # generate an array of random times for spiking to begin
        t[i, j] = random.random()    # t <= 1.0s
speed = np.full((num_units, 12), 5)  # speed was set arbitrarily at 5 mm/s so that t <= 1.0s
r = 5 - speed * t                    # 5mm chosen arbitrarily as the width of the projected image
# ****************************************


# ***************************** create the figure with appropriate subplots ******************************

# calculate numer of rows needed for the # of columns specified
num_rows = int(num_units // num_cols + math.ceil((num_units % num_cols) / num_cols))


unit_num = 0         # initialize counter in order to label subplots by unit number

plt.figure(figsize = (10, 12))

for row in range(num_rows):
    for col in range(num_cols):
        if unit_num < num_units:     # uses Gridspec and subplot2grid to create gridded array
            ax = plt.subplot2grid((num_rows, num_cols), (row, col), projection='polar')
            y = r[unit_num, :]       # select the row of the matrix corresponding to this unit
            ax.plot(theta, y)
            ax.set_rticks([])        # turn off radial ticks and labels
            ax.set_xticks([0, np.pi /2, np.pi, 3 * np.pi /2])
            ax.tick_params(axis='x', labelsize='x-small')  # get 4 small angle labels
            ax.set_rmax(5)     # scale all subplots to largest possible receptive field
            #ax.set_rmax(image_diam)   # scale all subplots to the image size
            ax.set_title(str(unit_num) + '      ', va = 'top', ha = 'right')
            ax.grid(True)
            unit_num += 1            # increment unit counter
        else:
            break

plt.tight_layout(w_pad=1.0)          # creates space between subplots

plt.show()


# In[ ]:
