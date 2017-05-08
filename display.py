"""
Name: display.py
Author: Ronald Kemker
Description: Various display and storing helper functions.

I need to fix the documentation in here.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numba import vectorize, float32, int32, float64, int64, uint16
from colorsys import rgb_to_hsv, hsv_to_rgb

@vectorize([int32(int32),
            int64(int64),
            float32(float32),
            float64(float64)])
def rgbToHSV(data):
    dataSize = data.shape
    output = np.zeros([np.prod(dataSize[0:2]),3])

    data = data.reshape([np.prod(dataSize[0:2]),-1])
    for i in range(0,np.prod(dataSize[0:2])):
        output[i,:] = rgb_to_hsv(data[i,0],data[i,1],data[i,2])
    
    return output.reshape(dataSize)

@vectorize([int32(int32),
        int64(int64),
        float32(float32),
        float64(float64)])
def hsvToRGB(data):
    dataSize = data.shape
    output = np.zeros([np.prod(dataSize[0:2]),3])

    data = data.reshape([np.prod(dataSize[0:2]),-1])
    for i in range(0,np.prod(dataSize[0:2])):
        output[i,:] = hsv_to_rgb(data[i,0],data[i,1],data[i,2])
    
    return output.reshape(dataSize)

@vectorize([int32(int32),
            int64(int64),
            float32(float32),
            float64(float64),
            uint16(uint16)])    
def histEq(data):
    dataSize = data.shape
    hist = np.histogram(data,bins=256)[0]
    p = hist/np.prod(dataSize)
    T = np.floor(255*np.cumsum(p))    
    return T[data.astype(int)]

def imshow(data, equalize=False):
    '''
    imshow - stretch color image to fit in Matplotlib imshow
    Input:
           data - numpy array (height, width, 3)
    '''    
    sh = data.shape
    if len(sh) > 2:    
        data = data.reshape(-1,3)
    else:
        data = data.ravel()
    data = MinMaxScaler().fit_transform(data)*255.0
    
    if equalize:
        data = histEq(data)
    
    plt.imshow(np.uint8(data.reshape(sh)))

def imsave(data, fName, equalize=False, dpi=600):    
    '''
    imsave - Save high-quality image
    Input:
           data - numpy array (height, width, 3)
    '''    
    sh = data.shape
    
    if len(sh) > 2:    
        data = data.reshape(-1,3)
    else:
        data = data.ravel()
        
    data = MinMaxScaler().fit_transform(data)*255.0
    
    if equalize:
        data = histEq(data)
    
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.uint8(data.reshape(sh)), aspect='auto')
    fig.savefig(fName,dpi=dpi)
    plt.close()

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def classmap_show(data, cmap=None, numClasses = None):

    if numClasses is None:
        numClasses = np.max(data)

    if cmap is None:
        cmap = discrete_cmap(numClasses)

    plt.imshow(data,cmap=cmap,vmin=0, vmax=numClasses-1)

def classmap_save(data, fname, cmap=None, numClasses=None, dpi=300):
    fig = plt.figure(frameon=False)
    data = np.uint8(data)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    if numClasses is None:
        numClasses = np.max(data)

    if cmap is None:
        cmap = discrete_cmap(numClasses)

    ax.imshow(data, aspect='auto', cmap=cmap,vmin=0, vmax=numClasses-1)
    fig.savefig(fname,dpi=dpi)
    plt.close()
