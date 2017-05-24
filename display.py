"""
Name: display.py
Author: Ronald Kemker
Description: Various display and image storing helper functions.  Very helpful
for figures in publications.

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
    """Convert image from RGB to HSV
    
    Parameters
    ----------
    data : numpy array [rows x columns x channels], input RGB image
    
    Returns
    -------
    output : numpy array [rows x columns x channels], output HSV image
    """
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
    """Convert image from HSV to RGB
    
    Parameters
    ----------
    data : numpy array [rows x columns x channels], input HSV image
    
    Returns
    -------
    output : numpy array [rows x columns x channels], output RGB image
    """
    dataSize = data.shape
    output = np.zeros([np.prod(dataSize[0:2]),3])

    data = data.reshape([np.prod(dataSize[0:2]),-1])
    for i in range(0,np.prod(dataSize[0:2])):
        output[i,:] = hsv_to_rgb(data[i,0],data[i,1],data[i,2])
    
    return output.reshape(dataSize)

def imshow(data):
    """imshow for arbitrary (non-UINT8) image
    
    Parameters
    ----------
    data : numpy array [rows x columns x channels], input RGB image
    """   
    sh = data.shape
    if len(sh) > 2:    
        data = data.reshape(-1,3)
    else:
        data = data.ravel()
    data = MinMaxScaler().fit_transform(data)*255.0
        
    plt.imshow(np.uint8(data.reshape(sh)))

def imsave(data, fName, dpi=600):    
    """Save figure with no border
    
    Parameters
    ----------
    data : numpy array [rows x columns x channels], input image
    fName : string, file path to save image
    dpi : int, dots-per-inch (Default: 600)
    
    """   
    sh = data.shape
    
    if len(sh) > 2:    
        data = data.reshape(-1,3)
    else:
        data = data.ravel()
        
    data = MinMaxScaler().fit_transform(data)*255.0
        
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

def classmap_show(data, cmap=None, num_classes = None):
    """Show a classification (semantic/instance segmentation) map

    Parameters
    ----------
    data : numpy array of integers [rows x columns], classification image
    cmap : custom colormap (Default : None -> Builds it from discrete_cmap)
    num_classes : integer, total number of distinct classes 
        (Default: None -> uses max value in input class map)
    """    
    if num_classes is None:
        num_classes = np.max(data)
    
    if cmap is None:
        cmap = discrete_cmap(num_classes)
    
    plt.imshow(data,cmap=cmap,vmin=0, vmax=num_classes-1)

def classmap_save(data, fname, cmap=None, num_classes=None, dpi=300):
    """Save a classification (semantic/instance segmentation) map

    Parameters
    ----------
    data : numpy array of integers [rows x columns], classification image
    fName : string, file path to save image
    cmap : custom colormap (Default : None -> Builds it from discrete_cmap)
    num_classes : integer, total number of distinct classes 
        (Default: None -> uses max value in input class map)
    dpi : int, dots-per-inch (Default: 300)
    """    
    fig = plt.figure(frameon=False)
    data = np.uint8(data)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    if num_classes is None:
        num_classes = np.max(data)

    if cmap is None:
        cmap = discrete_cmap(num_classes)

    ax.imshow(data, aspect='auto', cmap=cmap,vmin=0, vmax=num_classes-1)
    fig.savefig(fname,dpi=dpi)
    plt.close()
