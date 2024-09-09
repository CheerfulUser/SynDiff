import numpy as np 
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from tools import query_ps1, download_skycells
import pandas as pd
from copy import deepcopy

class ps1_data():
    def __init__(self,file,mask=False,catalog=None,toflux=True,pad=0):
        self.file = file 
        self.pad = pad
        self.ftype = 'log'   
        self.cat = None     
        self._load_image()
        self._load_mask(mask)
        self.get_catalog(catalog)

        if toflux:
            self.data = self.convert_flux_scale()
    
    def _load_image(self,file=None):
        print('')
        hdul = fits.open(self.file)
        if len(hdul) == 1:
            j = 0 
        else:
            j = 1
        self._bzero = deepcopy(hdul[j].header['BZERO'])
        self._bscale = deepcopy(hdul[j].header['BSCALE'])
        self.data = deepcopy(hdul[j].data)
        self.header = hdul[j].header
        self.wcs = WCS(hdul[j].header)

        hdul.close()
        self.im_skycell = int(self.file.split('skycell.')[-1].split('.')[0])
        self.band = self.file.split('stk.')[-1].split('.')[0]
        self.zp = 25 + 2.5*np.log10(self.header['EXPTIME'])
        self.ra, self.dec = self.wcs.all_pix2world(self.data.shape[1]/2,self.data.shape[0]/2,1)
        self.image_stats = sigma_clipped_stats(self.data)    
        self.padded = np.pad(self.data,self.pad)
    
    def _load_mask(self,mask):
        if mask:
            name = self.file.split('v3.')[-1].split('.stk')[0]
            download_skycells([name],self.file.split('rings')[0],filters=[self.band],overwrite=False,mask=True)
            mask_file = self.file.split('.fits')[0] + '.mask.fits'
            hdul = fits.open(mask_file)
            if len(hdul) == 1:
                j = 0 
            else:
                j = 1
            self._bscale_mask = deepcopy(hdul[j].header['BSCALE'])
            self._bzero_mask = deepcopy(hdul[j].header['BZERO'])
            self.mask = np.pad(hdul[j].data,self.pad)
            self.mask_header = hdul[j].header
            self.mask_file = mask_file
            hdul.close()
            if self.pad > 0:
                self.padded_mask = np.pad(self.mask,self.pad) 
        else:
            self.mask_header = None
            self.mask_file = None
            self.mask = None

        
    def convert_flux_scale(self,toflux=True):
        a = 2.5/np.log(10)
        if toflux:
            if self.ftype == 'log':
                self.ftype = 'flux'

                x = self.data/a
                flux = self.header['boffset'] + self.header['bsoften'] * 2 * np.sinh(x)
                self.data = flux 

                x = self.padded/a
                flux = self.header['boffset'] + self.header['bsoften'] * 2 * np.sinh(x)
                self.padded = flux 
        else:
            if self.ftype == 'flux':
                self.ftype = 'log'

                tmp = (self.data - self.header['boffset']) / (self.header['bsoften']*2)
                log = np.arcsinh(tmp)*a
                self.data = log

                tmp = (self.padded - self.header['boffset']) / (self.header['bsoften']*2)
                log = np.arcsinh(tmp)*a
                self.padded = log

    def save_image(self,savepath,savename,overwrite=False):
        self.convert_flux_scale(toflux=False)
        data = self.padded[self.pad:-self.pad,self.pad:-self.pad]
        hdu = fits.PrimaryHDU(data=data,header=self.header)
        hdu.scale('int16', bscale=self._bscale,bzero=self._bzero)
        hdul = fits.HDUList([hdu])
        savename = savepath + '/' + savename + '.fits' #self.file.split('fits')[0].split('/')[-1] + f'{suffix}.fits'
        hdul.writeto(savename,overwrite=overwrite)

    def save_mask(self,savepath,savename,overwrite=False):
        data = self.mask
        data = data[self.pad:-self.pad,self.pad:-self.pad]
        hdu = fits.PrimaryHDU(data=data,header=self.mask_header)
        hdu.scale('int16', bscale=self._bscale_mask,bzero=self._bzero_mask)
        hdul = fits.HDUList([hdu])
        savename = savepath + '/' + savename + '.fits'#self.mask_file.split('fits')[0].split('/')[-1] + f'{suffix}.fits'
        hdul.writeto(savename,overwrite=overwrite)

    def set_padding(self,pad):
        self.pad = pad
        self.padded = np.pad(self.data,pad)

    def get_catalog(self,catalog):
        if (self.cat is None) & (type(catalog) == bool):
            if catalog:
                cat = query_ps1(self.ra,self.dec,0.41)
            else:
                return
        elif catalog is not None:
            if type(catalog) == str:
                cat = pd.read_csv(catalog)
            else:
                cat = catalog 
        else:
            return

        x,y = self.wcs.all_world2pix(cat['raMean'].values,cat['decMean'].values,0)
        x += self.pad; y += self.pad
        cat['x'] = x
        cat['y'] = y
        ind = (x > 0) & (y > 0) & (x < self.padded.shape[1]) & (y < self.padded.shape[0])
        cat = cat.iloc[ind]
        cat = cat.loc[(cat['iMeanPSFMag'] > 0) & (cat['rMeanPSFMag'] > 0) & 
                      (cat['zMeanPSFMag'] > 0) & (cat['yMeanPSFMag'] > 0)]
        cat = cat.sort_values(self.band+'MeanPSFMag')
        cat = cat.drop_duplicates()
        self.cat = cat 
