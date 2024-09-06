from glob import glob
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.signal import fftconvolve

from astropy.modeling import models

from datetime import date
import traceback


from correct_saturation import saturated_stars,mask_rad_func
from correct_saturation_old import saturated_stars_old
from pad_skycell import pad_skycell
from ps1_data_handler import ps1_data

import warnings
# nuke warnings because sigma clip is extremely annoying 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class combine_ps1():
    def __init__(self,datapath,skycells,psf_std=70,combine=[0.238,0.344,0.283,0.135],
                 catalog_path=None,savepath='.',suffix='rizy.conv',
                 use_mask=True,overwrite=False,pad=500,verbose=0,run=True):
        self.datapath = datapath
        self.psf_std = psf_std
        self.combine = np.array(combine)
        self.skycells = skycells
        self.catalog_path = catalog_path
        self.savepath = savepath
        self.suffix = suffix
        self.overwrite = overwrite
        self.pad = pad
        self.verbose = verbose
        self.use_mask = use_mask
        if run:
            self.run()
        
        
    def run(self):
        self._gather_ps1()
        self._load_skycells()
        self._make_psf()
        self.process()
        
    def _gather_ps1(self):
        files = np.array(glob(f'{self.datapath}/*.unconv.fits'))
        cell = np.array([f.split('.stk')[0] + '.stk.' for f in files])
        good = []
        for i in range(len(cell)):
            if sum(cell[i] == cell) == 4:
                good += [cell[i]]
        good = list(set(good))
        self.fields = good
        #if self.skycell is not None:
    
    def _load_skycells(self):
        if type(self.skycells) == str:
            self.skycells = pd.read_csv(self.skycells)
    
    def _make_psf(self):
        size = 2000
        y, x = np.mgrid[:size, :size]
        x = x - size/2; y = y - size/2
        psfg = models.Gaussian2D(x_stddev=self.psf_std,y_stddev=self.psf_std)
        psfg = psfg(x,y)
        psfg /= np.nansum(psfg)
        self.psf = psfg
        
    
    def process(self):
        for file in self.fields:
                try:
                    out = self.savepath + file.split('/')[-1] + self.suffix + '.fits'
                    exist = glob(out)
                    if (len(exist) == 0) | self.overwrite:
                        if self.verbose > 0:
                            print(f'Starting field {file.split('/')[-1].split('.stk')[0]}')
                        bands = ['r','i','z','y']
                        images = []
                        masks = []
                        for b in bands:
                            f = file + f'{b}.unconv.fits'
                            if b == 'r':
                                ps1 = ps1_data(f,mask=self.use_mask,catalog=self.catalog_path+'ps1.csv',
                                               toflux=False,pad=self.pad)
                            else:
                                ps1._load_image(f)
                                ps1._load_mask(f)

                            pad = pad_skycell(ps1=ps1,skycells=self.skycells,datapath=self.datapath)
                            sat = saturated_stars(deepcopy(pad.ps1))#,catalogpath=self.catalog_path)
                            images += [sat.ps1.padded]
                            masks += [sat.ps1.mask]
                            print(f'Done {b}')
                        self.ps1 = sat.ps1
                        images = np.array(images)
                        masks = np.array(masks,dtype=int)
                        image = np.nansum(images*self.combine[:,np.newaxis,np.newaxis],axis=0)
                        image = fftconvolve(image,self.psf,mode='same')
                        mask = masks[0]
                        for m in masks[1:]:
                            mask = mask | m
                        self.ps1.padded = image
                        self.ps1.mask = mask
                        self._update_header()
                        savename = file.split('/')[-1] + f'{self.suffix}'
                        self.ps1.save_image(savepath,savename,overwrite=self.overwrite)
                        print('Saved: ',savename)
                        savename = file.split('/')[-1] + f'{self.suffix}.mask'
                        self.ps1.save_mask(savepath,savename,overwrite=self.overwrite)
                except Exception:
                    print(traceback.format_exc())

            
    def _update_header(self):
        header = self.ps1.header
        badkeys = ['HISTORY','INP_*','SCL_*','ZPT_*','EXP_*','AIR_*','HIERARCH*']
        for key in badkeys:
            del header[key]
        header['FILTER'] = ('rizy','Filter used')
        header['COMBINE'] = (True, 'Combined image')
        header['PSFTYPE'] = ('Gaussian', 'Type of PSF used in convolution')
        header['PSFstd'] = (self.psf_std, 'Standard deviation of Gaussian')
        header['FRACR'] = (self.combine[0], 'Fraction of r used')
        header['FRACI'] = (self.combine[1], 'Fraction of i used')
        header['FRACZ'] = (self.combine[2], 'Fraction of z used')
        header['FRACY'] = (self.combine[3], 'Fraction of y used')
        header['COMBDATE'] = (date.today().isoformat(),'Date of combination')
        
        self.ps1.header = header