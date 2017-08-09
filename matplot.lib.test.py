
# coding: utf-8

# In[204]:

import matplotlib.pyplot as plt
import numpy as np
t = np.arange(0,12,.01)
plt.plot(t,t**2,'k')
plt.ylabel('some numbers squared')
#plt.axis([0,5,0,25])
#plt.axes(polar = True)
plt.show()


# In[22]:

plt.setp(lines)


# In[24]:

lines = plt.plot(1,2)
plt.setp(lines)


# In[128]:


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(221)
plt.plot(t1, f(t1), 'bo', t2, f(t2))
plt.axis([0,5,-1,1])

plt.subplot(222)
plt.plot(t2, np.cos(2*np.pi*t2), '--')

plt.subplot(223)
plt.plot(t1, f(t1), 'go', t2, f(t2), 'k')
plt.axis([0,5,-1,1])

plt.subplot(2,2,4)
plt.plot(t2, np.cos(2*np.pi*t2), 'g--')

plt.show()


# In[57]:

plt.gca()
#plt.gcf()


# In[59]:

plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])


plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3') # subplot 211 title
plt.show()


# In[117]:


# Fixing random state for reproducibility
np.random.seed(19680801)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1,facecolor='b', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(62, .027, r'$\mu=100\ \sigma=15$', name = 'Arial')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


# In[114]:

rm ~/.cache/matplotlib/fontList.cache


# In[118]:

import matplotlib.patches as patches

# build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

# axes coordinates are 0,0 is bottom left and 1,1 is upper right
p = patches.Rectangle(
    (left, bottom), width, height,
    fill=False, transform=ax.transAxes, clip_on=False
    )

ax.add_patch(p)

ax.text(left, bottom, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, bottom, 'left bottom',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right bottom',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right top',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(right, bottom, 'center top',
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, 0.5*(bottom+top), 'right center',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, 0.5*(bottom+top), 'left center',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(0.5*(left+right), 0.5*(bottom+top), 'middle',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=20, color='red',
        transform=ax.transAxes)

ax.text(right, 0.5*(bottom+top), 'centered',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, top, 'rotated\nwith newlines',
        horizontalalignment='center',
        verticalalignment='center',
        rotation=45,
        transform=ax.transAxes)

ax.set_axis_off()
plt.show()


# In[159]:


# Create a new figure of size 8x6 points, using 100 dots per inch
plt.figure(figsize=(3,2), dpi=80)

# Create a new subplot from a grid of 1x1
plt.subplot(111)

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

# Plot cosine using blue color with a continuous line of width 1 (pixels)
plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-", rasterized = True, alpha = 0.9)

# Plot sine using green color with a continuous line of width 1 (pixels)
plt.plot(X, S, color="green", linewidth=1.0, linestyle="-", rasterized = True)

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
       [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks([-1, 0, +1],
       [r'$-1$', r'$0$', r'$+1$'])

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

# Save figure using 72 dots per inch
# savefig("../figures/exercice_2.png",dpi=72)

# Show result on screen
plt.show()


# In[201]:

# plt.figure creates a matplotlib.figure.Figure instance
fig = plt.figure()
rect = fig.patch # a rectangle instance
rect.set_facecolor('lightgoldenrodyellow')

ax1 = fig.add_axes([0.1, 0.3, 0.5, 0.5])
rect = ax1.patch
rect.set_facecolor('LAVENDER')


for label in ax1.xaxis.get_ticklabels():
    # label is a Text instance
    label.set_color('red')
    label.set_rotation(60)
    label.set_fontsize(12)

for line in ax1.xaxis.get_ticklines():
    line.set_markersize(6)
    line.set_markeredgewidth(2)

for line in ax1.yaxis.get_ticklines():
    # line is a Line2D instance
    line.set_color('green')
    line.set_markersize(6)
    line.set_markeredgewidth(2)

plt.show()


# In[286]:

X = np.arange(-np.pi, np.pi,.01)
Y, Z, A, B = np.sin(X), np.cos(X), np.sin(X+ np.pi/2 ), np.cos(X + np.pi/2)
fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(221)
plt.plot( X, Y, 'g-', rasterized = True, linewidth=2.0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.yticks([-1,0,1])
ax1.spines['left'].set_position(('axes',0.043))
ax1.spines['right'].set_position(('axes',0.957))
ax1.spines['top'].set_smart_bounds(True)
ax1.spines['bottom'].set_smart_bounds(True)
plt.ylabel('Volts', color = 'green')
#ax1.text(0, .7, 'Volts', color = 'green', rotation = 'vertical', horizontalalignment='left',verticalalignment='center', transform=ax.transAxes, fontsize = 12)


ax2 = fig.add_subplot(222)
plt.plot( X,Z,'g-', rasterized = True, linewidth=2.0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.yticks([-1,0,1])
ax2.spines['left'].set_position(('axes',0.043))
ax2.spines['right'].set_position(('axes',0.957))
ax2.spines['top'].set_smart_bounds(True)
ax2.spines['bottom'].set_smart_bounds(True)

#ax2.text(0, 0.7, 'Amps', color = 'blue', rotation = 'vertical', horizontalalignment='right',verticalalignment='center', transform=ax.transAxes, fontsize = 12 )
plt.ylabel('Amps', color = 'green')

ax3 = fig.add_subplot(223)
plt.plot( X, A, 'b-', rasterized = True, linewidth=2.0, label = 'Volts')
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.yticks([-1,0,1])
ax3.spines['left'].set_position(('axes',0.043))
ax3.spines['right'].set_position(('axes',0.957))
ax3.spines['top'].set_smart_bounds(True)
ax3.spines['bottom'].set_smart_bounds(True)
plt.ylabel('Volts', color = 'blue')
plt.legend(loc='upper left', frameon=True)

ax4 = fig.add_subplot(224)
plt.plot( X, B , 'b-', rasterized = True, linewidth=2.0, label = 'Amps')
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.yticks([-1,0,1])
ax4.spines['left'].set_position(('axes',0.043))
ax4.spines['right'].set_position(('axes',0.957))
ax4.spines['top'].set_smart_bounds(True)
ax4.spines['bottom'].set_smart_bounds(True)
plt.ylabel('Amps', color = 'blue')
#plt.legend(loc='upper left', frameon=False)

plt.show()


# In[517]:

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

plt.figure(figsize=(10,6), dpi=80)
plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
plt.plot(X, S, color="red",  linewidth=2.5, linestyle="-")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(C.min()*1.1, C.max()*1.1)

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
       [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks([-1, 0, +1],
       [r'$-1$', r'$0$', r'$+1$'])

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

t = 2*np.pi/3

plt.plot([t,t],[0,np.cos(t)], color ='blue', linewidth=2.5, linestyle="--")
plt.scatter([t,],[np.cos(t),], 50, color ='blue')

ax.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
             xy=(t, np.sin(t)), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

ax.plot([t,t],[0,np.sin(t)], color ='red', linewidth=2.5, linestyle="--")
ax.scatter([t,],[np.sin(t),], 50, color ='red')

ax.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
             xy=(t, np.cos(t)), xycoords='data',
             xytext=(-90, -50), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))


plt.show()


# In[317]:

import numpy as np
import matplotlib.pyplot as plt

plt.axes([0,0,1,1], polar = True)

N = 20
theta = np.arange(0.0, 2*np.pi, 2*np.pi/N)
radii = 10*np.random.rand(N)
width = np.pi/4*np.random.rand(N)
bars = plt.bar(theta, radii, width=width, bottom=0.0)

for r,bar in zip(radii, bars):
    bar.set_facecolor( 'blue')#cmap-jet.png(r/10.))
    bar.set_alpha(0.5)

plt.show()


# In[455]:

"""
Demo of a line plot on a polar axis.
"""
import numpy as np
import matplotlib.pyplot as plt


r = np.arange(0.01, 2, 0.01)
theta = 2 * np.pi * r

ax = plt.subplot(111, projection='polar')
ax.plot(theta, r)
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()


# In[360]:

#!/usr/bin/env python
# a polar scatter plot; size increases radially in this example and
# color increases with angle (just to verify the symbols are being
# scattered correctly).  In a real example, this would be wasting
# dimensionality of the plot
from pylab import *

N = 150
r = 2*rand(N)
theta = 2*pi*rand(N)
area = 200*r**2*rand(N)
colors = theta
ax = subplot(111, polar=True)
c = scatter(theta, r, c=colors,  s=area, cmap=cm.hsv)
c.set_alpha(0.75)

show()


# In[361]:

# needs a set of data to plot

def polar_plot(r, phi, data):
    """
    Plots a 2D array in polar coordinates.

    :param r: array of length n containing r coordinates
    :param phi: array of length m containing phi coordinates
    :param data: array of shape (n, m) containing the data to be plotted
    """
    # Generate the mesh
    phi_grid, r_grid = np.meshgrid(phi, r)
    x, y = r_grid*np.cos(phi_grid), r_grid*np.sin(phi_grid)
    plt.pcolormesh(x, y, data)
    plt.show()


# In[391]:

# first part of a cool graphing function from The Inertial Frame,
# a blog on data, graphing & python
# http://astro.cornell.edu/~pslii/?p=101

import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

plt.figure(figsize = (15,4))
x = np.arange(41)
ysize = len(x)
y = np.sin(toeplitz(x**1.3, np.zeros(ysize)))
plt.plot(x, y[:,40])

#plt.show()

coefficient = np.linspace(0.05, 1, ysize)
data = coefficient * y / y.max()


def plotLogo(x, data, ymin=-1, ymax=1):
    fig = plt.gcf()
    fig.set_size_inches(12,2)

    ax = plt.Axes(fig, [0,0,1,1], # fill the entire figure with the plot
                  yticks=[], xticks=[], # erase all tickmarks
                  frame_on=False, # turns off the black border around the plot
                  facecolor = 'white') # set background to white, change to '#000000' for black

    # set the limits of the plot
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(x.min(), x.max())

    # delete the original axes and add the clean one we just generated
    fig.delaxes(plt.gca())
    fig.add_axes(ax)

    # plot the actual data
    for i in range(ysize): plt.plot(x, data[:,i],
                                     color=str(1.0-(float(i)/ysize)), # fades the color from white to black
                                     lw=1.2, # set the linewidth
                                     alpha=0.2) # set the transparency: 0 is completely transparent, 1 is opaque
    plt.show()

plotLogo(x, data)


# In[542]:

"""
Plotter for Receptive Fields 7/14/17  Aaaaargh! Does not work! Back to drawing board!

Need to use a rectangular plot and some geometry to calculate dist. from "unit" location
to the moving bar stimulus.  Advantage is plots will show location of receptive field within
MEA in addition to size and shape.
"""
import numpy as np
import matplotlib.pyplot as plt
import random

# ******** fake data ******************
#theta = np.linspace(0, 2*np.pi, 12)
#t = [0.1, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1]
#t = np.reshape(t, 12, 1)
#speed = np.full(len(t),5)
#r = 5 - speed * t
# *************************************


plt.figure(figsize = (10, 12))

num_units = 47       # set equal to the number of units in data set
num_cols = 6         # choose based on desired size/spread of plots
image_diam = 5       # image size set arbitrarily to 5 mm

# calculate numer of rows needed for the # of columns specified
num_rows = int(num_units // num_cols + ceil((num_units % num_cols) / num_cols))
unit_num = 0         # initialize counter in order to label subplots by unit number

# ************ new randomized fake data *************
theta = np.linspace(0, 2*np.pi, 12)  # generates 12 evenly spaced angles; these are NOR correct
t = np.empty((num_units, 12))
for i in range(num_units):
    for j in range(12):              # generate an array of random times for spiking to begin
        t[i, j] = random.random() * 6   # t <= 5.0s
speed = np.full((num_units, 12), 1)  # speed was set arbitrarily at 1mm/s so that t <= 5.0s

r = 5 - speed * t                    # 5mm chosen arbitrarily as the width of the projected image
print(r[0:5])
# ****************************************

for row in range(num_rows):
    for col in range(num_cols):
        if unit_num < num_units:     # uses Gridspec and subplot2grid to create gridded array
            ax = plt.subplot2grid((num_rows, num_cols), (row, col), projection='polar')
            y = r[unit_num, :]       # select the row of the matrix corresponding to this unit
            ax.plot(theta, y)
            ax.set_rticks([])        # turn off radial ticks and labels
            ax.set_xticks([0, np.pi /2, np.pi, 3 * np.pi /2])
            ax.tick_params(axis='x', labelsize='x-small')  # get 4 small angle labels
            #ax.set_rmax(r.max())     # scale all subplots to largest receptive field
            ax.set_rmax(image_diam)   # scale all subplots to the image size            ax.set_title(str(unit_num) + '      ', va = 'top', ha = 'right')
            ax.grid(True)
            ax.set_title(str(unit_num) + '      ', va = 'top', ha = 'right')
            unit_num += 1            # increment unit counter
        else:
            break

plt.tight_layout(w_pad=1.0)          # creates space between subplots
plt.show()


# In[532]:

# first attempt to plot data similar to that generated for mapping receptive fields of neurons
# 12 data points showing the location of the bar at each angle when the cell began to fire
# (this could also be a fuzzy line showing results from multiple trials/uncertainty)


# ******** fake data ******************
theta = np.linspace(0, 2*np.pi, 12)
t = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 1.6, 0.7, 0.4, 0.2, 0.1]
t = np.reshape(t, 12, 1)
speed = np.full(len(t),5)
r = 5 - speed * t
# *************************************
plt.figure(figsize = (8.5, 11)) # not clear that setting figsize actually does anything ...
num_plots = 22 # set equal to the number of units in data set
num_cols = 6 # choose based on desired size/spread of plots
# calculate numer of rows needed for the # of columns specified
num_rows = num_plots // num_cols + num_plots % num_cols
for i in range(num_plots):
    ax = plt.subplot(num_rows, num_cols, i + 1, polar = True, alpha = 0.1)
    width = 2 * np.pi / 12
    bars = plt.bar(theta, r , width=width, bottom=0.0)
    ax.set_xticks([0, np.pi /2, np.pi, 3 * np.pi /2])
    ax.tick_params(axis='x', labelsize='xx-small')
    ax.set_rmax(r.max())
    ax.set_title(str(i) + '      ', va = 'top', ha = 'right')

#ax = plt.subplot(111, projection='polar')
#ax.plot(theta, r)


plt.show()



# In[420]:

# question - why are the bars appearing in "random" order instead of by theta?!
# Answer: 'projection = polar' uses radians instead of degrees!!
matplotlib.get_cachedir()


# In[428]:

csfont = {'fontname':'Kinnari-BoldOblique'}
hfont = {'fontname':'Helvetica'}

plt.title('title',**csfont)
plt.xlabel('xlabel', **hfont)
plt.show()


# In[426]:

from matplotlib import font_manager
print(font_manager.findSystemFonts(fontpaths=None))


# In[432]:

from matplotlib.font_manager import FontProperties, findfont
fp = FontProperties(family='comic',
                    style='normal',
                    variant='normal',
                    weight='normal',
                    stretch='normal',
                    size='medium')

font = findfont(fp)
print(font)


# In[525]:

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x**2)
fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
axes[0, 0].plot(x, y)
axes[1, 1].scatter(x, y)
plt.show()


# In[556]:

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np

fig,ax = plt.subplots(1)

N = 5
nfloors = np.random.rand(N) # some random data

patches = []

cmap = plt.get_cmap('RdYlBu')
colors = cmap(nfloors) # convert nfloors to colors that we can use later

for i in range(N):
    verts = np.random.rand(3,2)+i # random triangles, plus i to offset them
    polygon = Polygon(verts,closed=True)
    patches.append(polygon)

collection = PatchCollection(patches)

ax.add_collection(collection)

collection.set_color(colors)
collection.set_alpha(0.25)

ax.autoscale_view()
plt.show()


# In[559]:

print(np.sin(np.pi/2)==np.sin(2*np.pi-np.pi/2))


# In[573]:

life =[2.889, 3.007,4.009, 6.014, 2.005, 12.028, 1.718, 4.009, 2.005, 1.503, 12.028, 1.503, 2.406, 1.503, 1.503, 1.503]
speed = [500, 400, 300, 200, 600, 100, 700, 300, 600, 800, 100, 800, 500, 800, 800, 800]
angle = [3.4, 5.49, 4.45, 4.45, 2.88, 6.021, 3.4, 6.02, 5.498, 4.45, 0.262, 0.785, 2.88, 5.5, 2.88, 4.97]

print(len(life), len(speed), len(angle))


# In[577]:

#plt.scatter(speed, life, color = 'green')
plt.scatter(angle, life, color = 'blue')
plt.show()


# In[599]:

import math

math.sin(1.57)


# In[1]:

abs(-1)


# In[ ]:
