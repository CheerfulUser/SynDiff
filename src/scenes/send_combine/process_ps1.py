from glob import glob
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.signal import fftconvolve

import sys

from astropy.modeling import models
from astropy.io import fits

from datetime import date
import traceback

from tqdm import tqdm

from correct_saturation import saturated_stars,mask_rad_func
# from correct_saturation import saturated_stars
from pad_skycell import pad_skycell
from ps1_data_handler import ps1_data
from tools import _save_space

from joblib import Parallel, delayed

import warnings
# nuke warnings because sigma clip is extremely annoying 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def save_image(data, header, savepath,savename,overwrite=False):
    hdu = fits.PrimaryHDU(data=data,header=header)
    hdul = fits.HDUList([hdu])
    if savepath[-1] == '/':
        savename = savepath + savename + '.fits'
    else:
        savename = savepath + '/' + savename + '.fits' #self.mask_file.split('fits')[0].split('/')[-1] + f'{suffix}.fits'
    hdul.writeto(savename,overwrite=overwrite)

def save_mask(mask, mask_header, savepath,savename,overwrite=False):
    hdu = fits.PrimaryHDU(data=mask,header=mask_header)
    # hdu.scale('int16', bscale=self._bscale_mask,bzero=self._bzero_mask)
    hdul = fits.HDUList([hdu])
    if savepath[-1] == '/':
        savename = savepath + savename + '.fits'
    else:
        savename = savepath + '/' + savename + '.fits' #self.mask_file.split('fits')[0].split('/')[-1] + f'{suffix}.fits'
    hdul.writeto(savename,overwrite=overwrite)

class combine_ps1():
    def __init__(self,datapath,skycells,psf_std=70,combine=[0.238,0.344,0.283,0.135],
                 catalog_path=None,savepath='.',suffix='rizy.conv',
                 use_mask=True,overwrite=False,pad=500,verbose=0,run=True,cores=5):
        self.datapath = datapath
        self.psf_std = psf_std
        self.combine = np.array(combine)
        self.skycells = skycells
        self.catalog_path = catalog_path
        
        # ps1_csv_name = catalog_path + '_skycell.' + str(skycell_id) + '_ps1.csv'
        ps1_csv_name = catalog_path + '_ps1.csv'
        # general_csv_name = combiner.catalog_path + '_skycell.' + str(skycell_id)
        # catalog = pd.read_csv(ps1_csv_name)
        self.catalog = ps1_csv_name
        
        self.savepath = savepath
        self.check_savepath()
        self.suffix = suffix
        self.overwrite = overwrite
        self.pad = pad
        self.verbose = verbose
        self.use_mask = use_mask
        self.cores = cores
        if run:
            self.run()
        
        
    def run(self):
        self._load_skycells()
        
        self.csv_skycells = np.array([self.skycells['Name'][i].split('skycell.')[-1] for i in range(len(self.skycells))])
        
        self._gather_ps1()
        
        # sys.exit(0)
        self._make_psf()
        self.process()
        
    def check_savepath(self):
        _save_space(self.savepath)

    def _gather_ps1(self):
        files = np.array(glob(f'{self.datapath}/*.unconv.fits'))
        cell = np.array([f.split('.stk')[0] + '.stk.' for f in files])
        
        sc_ids = np.array([f.split('/')[-1].split('.stk')[0].split('skycell.')[-1] for f in files])
        
        good = []
        skycell_ids = []
        
        for i in range(len(cell)):
            if (sum(cell[i] == cell) >= 4):
                if (sc_ids[i] in self.csv_skycells):
                    good += [cell[i]]
                    skycell_ids += [sc_ids[i]]
        good = list(set(good))
        skycell_ids = list(set(skycell_ids))
        
        # print(len(skycell_ids), 'skycells found in the data.')
        
        self.csv_skycells
        
        if self.overwrite:
            self.fields = good
            self.skycell_ids = skycell_ids
        else:
            unique_skycell_indexes = self._lose_skycells(skycell_ids)
            self.fields = list(np.array(good)[~unique_skycell_indexes])
            self.skycell_ids = list(np.array(skycell_ids)[~unique_skycell_indexes])
        
    def _lose_skycells(self, skycell_ids):
        
        files = np.array(glob(f'{self.savepath}/*conv.mask.fits'))
        sc_ids = np.array([f.split('/')[-1].split('.stk')[0].split('skycell.')[-1] for f in files])
        unique_skycell_indexes = np.isin(skycell_ids, sc_ids)
        return unique_skycell_indexes
    
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
        if (self.cores != 1) & (self.cores != 0):
            Parallel(n_jobs=self.cores)(delayed(_parallel_process)(self, i) for i in tqdm(range(len(self.fields)), desc="Processing fields"))
        else:
            self._process()

    def _process(self):
        
        for ind in tqdm(range(len(self.fields)), desc="Processing fields"):
            file = self.fields[ind]
            skycell_id = self.skycell_ids[ind]
        
        # for file in self.fields:
            try:
                out = self.savepath + file.split('/')[-1] + self.suffix + '.fits'
                # print(out)
                # print(self.savepath)
                exist = glob(out)
                if (len(exist) == 0) | self.overwrite:
                    # if self.verbose > 0:
                        # print(f"Starting field {file.split('/')[-1].split('.stk')[0]}")
                    bands = ['r','i','z','y']
                    images = []
                    masks = []
                    
                    catalog = deepcopy(self.catalog)
                    
                    for b in bands:
                        f = file + f'{b}.unconv.fits'
                        
                        ps1 = ps1_data(f,mask=self.use_mask,catalog=catalog,
                                        toflux=True,pad=self.pad)
                        # print('ZZZ', ps1.band)
                        # print('ZZZ', ps1.padded)
                        # print('ZZZ', ps1.padded.shape)

                        pad = pad_skycell(ps1=ps1,skycells=self.skycells,datapath=self.datapath)
                        
                        sat = saturated_stars(deepcopy(pad.ps1), catalog_ps1 = catalog, catalogpath = self.catalog_path)#, catalogpath=ps1_csv_name)
                        # sat.ps1.convert_flux_scale(toflux=True)
                        images += [sat.ps1.padded]
                        masks += [sat.ps1.mask]
                        # print(f'Done {ps1.band}')
                    self.ps1 = sat.ps1
                    images = np.array(images)
                    masks = np.array(masks,dtype=int)
                    image = np.nansum(images*self.combine[:,np.newaxis,np.newaxis],axis=0)
                    image = fftconvolve(image,self.psf,mode='same')
                    mask = masks[0]
                    for m in masks[1:]:
                        mask = mask | m
                    header = self._update_header()
                    mask_header = deepcopy(header)
                    mask_header['MASK'] = (True, 'Mask used in combination')
                    
                    savename = file.split('/')[-1] + f'{self.suffix}'
                    save_image(image, header, self.savepath,savename,overwrite=self.overwrite)
                    savename = file.split('/')[-1] + f'{self.suffix}.mask'
                    # self.ps1.save_mask(self.savepath,savename,overwrite=self.overwrite)
                    save_mask(mask, mask_header, self.savepath,savename,overwrite=self.overwrite)
            except Exception:
                # print(traceback.format_exc())
                raise ValueError(f"Error processing file {file}. Check the input data and parameters.")

            
    def _update_header(self):
        header = self.ps1.header
        badkeys = ['HISTORY','INP_*','SCL_*','ZPT_*','EXP_*','AIR_*','HIERARCH*', 'BSOFTEN', 'BOFFSET']
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
        header['LINEAR'] = (True, 'Linearly Combined image')
        
        # self.ps1.header = header
        
        return header

def _parallel_process(combiner,ind):
    
    file = combiner.fields[ind]
    skycell_id = combiner.skycell_ids[ind]
    
    try:
        out = combiner.savepath + file.split('/')[-1] + combiner.suffix + '.fits'
        exist = glob(out)
        if (len(exist) == 0) | combiner.overwrite:
            # if combiner.verbose > 0:
                # print(f"Starting field {file.split('/')[-1].split('.stk')[0]}")
            bands = ['r','i','z','y']
            images = []
            masks = []
            
            catalog = deepcopy(combiner.catalog)
            
            for b in bands:
                f = file + f'{b}.unconv.fits'
                # print(f)
                
                ps1 = ps1_data(f,mask=combiner.use_mask,catalog=catalog,
                               toflux=True,pad=combiner.pad)

                pad = pad_skycell(ps1=ps1,skycells=combiner.skycells,datapath=combiner.datapath)
                sat = saturated_stars(deepcopy(pad.ps1), catalog_ps1 = catalog, catalogpath = combiner.catalog_path)
                images += [sat.ps1.padded]
                masks += [sat.ps1.mask]
                # print(f'Done {ps1.band}')
            combiner.ps1 = sat.ps1
            images = np.array(images)
            masks = np.array(masks,dtype=int)
            image = np.nansum(images*combiner.combine[:,np.newaxis,np.newaxis],axis=0)
            image = fftconvolve(image,combiner.psf,mode='same')
            mask = masks[0]
            for m in masks[1:]:
                mask = mask | m
            # combiner.ps1.padded = image
            # combiner.ps1.mask = mask
            header = combiner._update_header()
            mask_header = deepcopy(header)
            mask_header['MASK'] = (True, 'Mask used in combination')
            
            savename = file.split('/')[-1] + f'{combiner.suffix}'
            save_image(image, header, combiner.savepath,savename,overwrite=combiner.overwrite)
            # print('Saved: ',savename)
            savename = file.split('/')[-1] + f'{combiner.suffix}.mask'
            # combiner.ps1.save_mask(combiner.savepath,savename,overwrite=combiner.overwrite)
            # print('Saving: ',savename)
            save_mask(mask, mask_header, combiner.savepath, savename, overwrite=combiner.overwrite)
    except Exception:
        # print(traceback.format_exc())
        raise ValueError(f"Error processing file {file}. Check the input data and parameters.")
