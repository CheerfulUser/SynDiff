#%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import sys
sys.path
sys.path.append('../SynDiff/scenes/')

import syndiff as sd 
from delta_kernal import *

import lightkurve as lk
import pandas as pd
datapath = '/Volumes/NZ_BACKUP/tess/cutouts/'

from glob import glob

import sys
sys.path.append('../Sigma_clip/')
import sigmacut

from astropy.wcs import WCS

def sigma_mask(data,error= None,sigma=3,Verbose= False):
    if type(error) == type(None):
        error = np.zeros(len(data))
    
    calcaverage = sigmacut.calcaverageclass()
    calcaverage.calcaverage_sigmacutloop(data,Nsigma=sigma
                                         ,median_firstiteration=True,saveused=True)
    if Verbose:
        print("mean:%f (uncertainty:%f)" % (calcaverage.mean,calcaverage.mean_err))
    return calcaverage.clipped

from astropy.io import fits

from tqdm.notebook import tqdm
from copy import deepcopy

from scipy.signal import savgol_filter


def Column_mask(files):
    summed = []
    for i in range(len(files)):
        hdu = fits.open(files[i])
        data = hdu[1].data
        summed += [np.nansum(data,axis=1)]
    summed = np.array(summed)

    lim = np.percentile(summed,1,axis=0)
    ind = summed < lim
    ind = np.where(ind)
    ref = np.zeros((len(files),data.shape[0],data.shape[1])) #np.nanmedian(data)
    for i in range(len(files)):
        if i in ind[0]:
            hdu = fits.open(files[i])
            data = hdu[1].data
            inds =ind[1][ind[0] == i]
            ref[i:i+1,:,inds] = data[np.newaxis,:,inds]
    ref[ref== 0] = np.nan
    ref = np.nanmedian(ref,axis=0)
    mask = ((ref <= np.percentile(ref,40,axis=0)) |
           (ref <= np.percentile(ref,10))) * 1.0
    return mask

def Boarder_caps(masked):
    jj = [0,1,-2,-1]
    for j in jj: 
        x = np.arange(0,masked.shape[1])
        y = masked[j,:].copy()
        finite = np.isfinite(y)
        if len(y[finite]) > 5:
            bad = sigma_mask(y[finite],sigma=3)
            finite = np.where(finite)[0]
            y[finite[bad]] = np.nan
            finite = np.isfinite(y)
            regressionLine = np.polyfit(x[finite], y[finite], 3)
            p = np.poly1d(regressionLine)

            thingo = y - p(x)
            finite = np.isfinite(thingo)
            bad = sigma_mask(thingo[finite],sigma=2)
            finite = np.where(finite)[0]
            y[finite[bad]] = np.nan
            finite = np.isfinite(y)
            regressionLine = np.polyfit(x[finite], y[finite], 3)
            p = np.poly1d(regressionLine)
            masked[j,np.isnan(masked[j,:])] = p(x)[np.isnan(masked[j,:])]
    return masked




def Column_background(data, cap = True):
    bkg = np.zeros_like(data)
    if cap:
        masked = Boarder_caps(data)
    for j in tqdm(range(data.shape[1])):
        x = np.arange(0,data.shape[1])
        y = data[:,j].copy()
        finite = np.isfinite(y)
        if len(y[finite]) > 5:
            finite = np.isfinite(y)
            regressionLine = np.polyfit(x[finite], y[finite], 10)
            p = np.poly1d(regressionLine)
            p =savgol_filter(p(x),25,3)

            thingo = y - p
            finite = np.isfinite(thingo)
            bad = sigma_mask(thingo[finite],sigma=2)
            finite = np.where(finite)[0]
            y[finite[bad]] = np.nan
            finite = np.isfinite(y)
            regressionLine = np.polyfit(x[finite], y[finite], 10)
            p = np.poly1d(regressionLine)
            p = savgol_filter(p(x),25,3)
            bkg[:,j] = p
    return bkg



path = '/Volumes/NZ_BACKUP/tess/s23/'
files = glob(path + '*.fits')
hdu = fits.open(files[10])
cs = hdu[1].header['SCCSA']-1 
ce = hdu[1].header['SCCED']-1 
rs = hdu[1].header['SCIROWS']-1 
re = hdu[1].header['SCIROWE']-1 

data = hdu[1].data#[rs:re,cs:ce]
wcs = WCS(hdu[1].header)
cut = Cutout2D(data,(1024+46,1024),2048,wcs=wcs)
cut = Cutout2D(cut.data,(750,1298),1500,wcs=cut.wcs)
data = cut.data
wcs = cut.wcs


for file in files:
    hdu = fits.open(file)
    data = hdu[1].data
    masked = data * mask
    bkg1 = Column_background(masked,False)
    bkg2 = Column_background(masked,True)
    res1 = np.nansum(abs(data - bkg1),axis=0)
    res2 = np.nansum(abs(data - bkg2),axis=0)
    diff = res2 - res1
    bkg = bkg2.copy()
    ind = diff > (np.nanmedian(diff)+np.nanstd(diff))
    bkg[:,ind] = bkg1[:,ind]
    
    test = deepcopy(hdu)
    test[0].data = test[0].data - bkg
    save = '/Volumes/NZ_BACKUP/tess/cutouts/reduced/'
    name = save + file.split('/')[-1]
    test.writeto(name,overwrite=True)
    hdu.close()
    test.close()