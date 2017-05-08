"""
Name: utils.py
Author: Ronald Kemker
Description: Helper functions for remote sensing applications.

Note:
Requires SpectralPython and GDAL
http://www.spectralpython.net/
http://www.gdal.org/
"""

import numpy as np
import spectral.io.envi as envi
from spectral import BandResampler
import gdal
from glob import glob
from random import sample
from numba import jit

def band_resample_hsi_cube(data, bands1, bands2, fwhm1,fwhm2 , mask=None):
    """
    band_resample_hsi_cube : Resample hyperspectral image

	Parameters
	----------    
	data : numpy array (height x width x spectral bands)
    bands1 : numpy array [1 x num source bands], 
		the band centers of the input hyperspectral cube
    bands2 : numpy array [1 x num target bands], 
		the band centers of the output hyperspectral cube
    fwhm1  : numpy array [1 x num source bands], 
		the full-width half-max of the input hyperspectral cube
    fwhm2 : numpy array [1 x num target bands],
		the full-width half-max of the output hyperspectral cube
	mask : numpy array (height x width), optional mask to perform the band-
		resampling operation on.

    Returns
    -------
	output - numpy array (height x width x N)

    """
    resample = BandResampler(bands1,bands2,fwhm1,fwhm2)    
    dataSize = data.shape
    data = data.reshape((-1,dataSize[2]))
    
    if mask is None:
        out = resample.matrix.dot(data.T).T
    else:
        out = np.zeros((data.shape[0], len(bands2)))
        mask = mask.ravel()
        out[mask] = resample.matrix.dot(data[mask].T).T
        
    out[np.isnan(out)] = 0
    return out.reshape((dataSize[0],dataSize[1],len(bands2)))

    
def readENVIHeader(fName):
    """
    Reads envi header

    Parameters
    ----------
    fName : String, Path to .hdr file
    
    Returns
    -------
    centers : Band-centers
	fwhm : full-width half-maxes
    """
    hdr = envi.read_envi_header(fName)
    centers = np.array(hdr['wavelength'],dtype=np.float)
    fwhm = np.array(hdr['fwhm'],dtype=np.float)
    return centers, fwhm
    
def read_hyperion_HSI_data(fileDir, dtype=np.float32):
    """
    Builds Hyperion HSI Cube from Directory

    Parameters
    ----------
    fileDir : String, Directory containing Hyperion imagery
	dtype : numpy dtype, Data type of output (Default: np.float32)
    
    Returns
    -------
    output : Numpy array - Calibrated radiance cube
    """
    dataFN = sorted(glob(fileDir+'/*.TIF'))
    calIdx = np.hstack([np.arange(7,57),np.arange(76,224)])              

    tmp = gdal.Open(dataFN[calIdx[0]]).ReadAsArray()
    
    data = np.zeros((tmp.shape[0],tmp.shape[1],198),dtype=dtype) 

    data[:,:,0] = tmp
    for i in range(1,198):
        data[:,:,i] = gdal.Open(dataFN[calIdx[i]]).ReadAsArray()
   
    data[:,:,0:50] = data[:,:,0:50]/40
    data[:,:,50:] = data[:,:,50:]/80

    return data

@jit
def patch_extractor_with_mask(data, mask, num_patches=50, patch_size=16):
    """
    Extracts patches inside a mask.  I need to find a faster way of doing this.

    Parameters
    ----------
    data : 3-D input numpy array [rows x columns x channels] 
	mask : 2-D binary mask where 1 is valid, 0 else
	num_patches : int, number of patches to extract (Default: 50)
    patch_size : int, pixel dimension of square patch (Default: 16)

    Returns
    -------
    output : 4-D Numpy array [num patches x rows x columns x channels]
	"""    
    sh = data.shape
    patch_arr = np.zeros((num_patches,patch_size, patch_size, sh[-1]), 
                         dtype=data.dtype)
    
    #Find Valid Regions to Extract Patches
    valid = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(0, sh[0]-patch_size):
        for j in range(0, sh[1]-patch_size):
            if np.all(mask[i:i+patch_size, j:j+patch_size]) == True:
                valid[i,j] += 1
    
    idx = np.argwhere(valid > 0 )
    idx = idx[np.array(sample(range(idx.shape[0]), num_patches))]
    
    for i in range(num_patches):
        patch_arr[i] = data[idx[i,0]:idx[i,0]+patch_size, 
                idx[i,1]:idx[i,1]+patch_size]
        
    return patch_arr    
    
