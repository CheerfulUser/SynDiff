from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

from scipy.optimize import minimize 
from copy import deepcopy
from astropy.stats import sigma_clipped_stats
from scipy.signal import fftconvolve

from tools import query_ps1, ps_psf, psf_minimizer, psf_phot, mask_rad_func
from ps1_data_handler import ps1_data

from photutils.aperture import CircularAperture
from photutils.aperture import ApertureStats, aperture_photometry


def image2counts(data,header,toflux=True):
	a = 2.5/np.log(10)
	if toflux:
		x = data/a
		flux = header['boffset'] + header['bsoften'] * 2 * np.sinh(x)
		return flux
	else:
		tmp = (data - header['boffset']) / (header['bsoften']*2)
		log = np.arcsinh(tmp)*a
		return log

class saturated_stars():
	def __init__(self,ps1,savepath,catalog=None,mask=False,
				 satlim=14,calmags=[15,17],pad=0,run=True,overwrite=False):
		"""
		"""
		self.ps1 = ps1
		self._load_ps1(mask)
		self.savepath = savepath
		self.overwrite = overwrite
		self.satlim = satlim
		self.calmags = calmags
		self._get_catalog(catalog)

		if run:
			self.fit_psf()
			self.flux_offset()
			self.replace_saturation()
			self._update_image()
			if self.ps1.mask is not None:
				self._update_mask()



	def _load_ps1(self,mask):
		if type(self.ps1) == str:
			self.ps1 = ps1_data(self.ps1,mask=mask)
		self.ps1.convert_flux_scale(toflux=True)


	def _get_catalog(self,catalog=None):
		self.ps1.get_catalog(catalog)

		self.ps1cat = self.ps1.cat
		self.ps1satcat = deepcopy(self.ps1cat.loc[self.ps1cat[self.ps1.band+'MeanPSFMag'] < self.satlim])
		self.calcat = self.ps1cat.loc[(self.ps1cat[self.ps1.band+'MeanPSFMag'] > self.calmags[0]) & (self.ps1cat[self.ps1.band+'MeanPSFMag'] < self.calmags[1])]



	def fit_psf(self,size=15):
		cal = self.calcat
		psf_mod = []
		print('psf fitting ',len(cal))
		for j in range(len(cal[:20])):
			cut, x, y = self._create_cut(cal.iloc[j],size)
			x0 = [10,10,0,1,0,0]
			res = minimize(psf_minimizer,x0,args=(cut,x,y))
			psf_mod += [res.x]
		psf_mod = np.array(psf_mod)
		psf_param = np.nanmedian(psf_mod,axis=0)
		self.psf_param = psf_param

	def flux_offset(self,size=15):
		cal = self.calcat
		fit_flux = []
		for j in range(len(cal)):
			cut, x, y = self._create_cut(cal.iloc[j],size)
			psf = ps_psf(self.psf_param[:-2],x-self.psf_param[-2],y-self.psf_param[-1])
			cflux = 10**((cal[self.ps1.band+'MeanPSFMag'].values[j]-self.ps1.zp)/-2.5)
			f = minimize(psf_phot,cflux,args=(cut,psf))
			fit_flux += [f.x]
		fit_flux = np.array(fit_flux)
		factor = np.nanmedian(fit_flux / 10**((cal[self.ps1.band+'MeanPSFMag'].values-self.ps1.zp)/-2.5))
		self.flux_factor = factor
		self._fitflux = fit_flux

	def _create_cut(self,source,size):
		xx = source['x']; yy = source['y']
		yint = int(yy+0.5)
		xint = int(xx+0.5)
		cut = deepcopy(self.ps1.padded[yint-size:yint+size+1,xint-size:xint+size+1])
		y, x = np.mgrid[:cut.shape[0], :cut.shape[1]]
		x = x - cut.shape[1] / 2 - (xx-xint)
		y = y - cut.shape[0] / 2 - (yy-yint)
		return cut,x,y


	def replace_saturation(self):
		masking = deepcopy(self.ps1.padded)
		inject = np.zeros_like(self.ps1.padded)
		sat = self.ps1satcat
		rads = mask_rad_func(sat[self.ps1.band+'MeanPSFMag'].values)
		cfluxes = 10**((sat[self.ps1.band+'MeanPSFMag'].values-self.ps1.zp)/-2.5)
		for i in range(len(sat)):
			rad = rads[i]
			cflux = cfluxes[i]
			y,x = np.mgrid[:rad*2,:rad*2]
			psf = ps_psf(self.psf_param[:-2],x-self.psf_param[-2]-rad,y-self.psf_param[-1]-rad)
			
			xx = sat['x'].values[i]; yy = sat['y'].values[i]
			dist = np.sqrt(((x-rad))**2 + ((y-rad))**2)
			ind = np.array(np.where(dist < rad))
			ind[0] += int(yy) - rad
			ind[1] += int(xx) - rad
			good = (ind[0] >= 0) & (ind[1] >= 0) & (ind[0] < self.ps1.padded.shape[0]) & (ind[1] < self.ps1.padded.shape[1])
			ind = ind[:,good]
			masking[ind[0],ind[1]] = np.nan
			dx = min(ind[1])
			dy = min(ind[0])
			
			inject[ind[0],ind[1]] += psf[ind[0]-dy,ind[1]-dx] * cflux*self.flux_factor + self.ps1.image_stats[1]
		if self.ps1.mask is not None:
			masking[self.ps1.mask>0] = self.ps1.image_stats[1] # set to the median
			self.newmask = inject > 0
		replaced = np.nansum([masking,inject],axis=0)
		self.replaced = replaced
		self.inject = inject


	def _update_image(self):
		newheader = deepcopy(self.ps1.header)
		newheader['SATCOR'] = (True,'Corrected sat stars')
		newheader['SATDATE'] = (date.today().isoformat(),'Date of correction')
		self.ps1.header = newheader
		self.ps1.padded = self.replaced
		#data = image2counts(self.replaced,self.ps1.header,toflux=False)
		#hdu = fits.PrimaryHDU(data=data,header=newheader)
		#hdul = fits.HDUList([hdu])
		#if self.savepath is not None:
		#	savename = self.savepath + self.ps1.file.split('fits')[0].split('/')[-1] + 'satcor.fits'
		#	hdul.writeto(savename,overwrite=self.overwrite)

	def _update_mask(self):
		newheader = deepcopy(self.ps1.mask_header)
		newheader['SATCOR'] = (True,'Corrected sat stars')
		#newheader['SATBIT'] = (0x8000,'Bit for the saturation mask')
		newheader['SATDATE'] = (date.today().isoformat(),'Date of correction')
		m = deepcopy(self.newmask).astype(int)
		m[m > 0] = 0x0020
		data = self.ps1.mask | m
		suspect = (fftconvolve(self.newmask, np.ones((240,240)), mode='same') > 0.5).astype(int)
		suspect[suspect] = 0x0080
		data = data | suspect
		self.ps1.mask = data
		self.ps1.mask_header = newheader
		#hdu = fits.PrimaryHDU(data=data,header=newheader)
		#hdul = fits.HDUList([hdu])
		#if self.savepath is not None:
		#	savename = self.savepath + self.ps1.mask_file.split('fits')[0].split('/')[-1] + 'satcor.fits'
		#	hdul.writeto(savename,overwrite=self.overwrite)

	def plot_catvspsf(self):
		plt.figure()
		plt.plot(self.calcat[self.ps1.band+'MeanPSFMag'],self._fitflux,label='PSF flux')
		plt.plot(self.calcat[self.ps1.band+'MeanPSFMag'],10**((self.calcat[self.ps1.band+'MeanPSFMag'].values-self.ps1.zp)/-2.5)*self.flux_factor,label='Corrected catalog flux')
		plt.xlabel(self.ps1.band+' mag')
		plt.ylabel('Counts')
		plt.legend()
		plt.title('Correction: ' + str(np.round(self.flux_factor,2))+r'$\times$flux')

	def plot_result(self):
		vmin=np.nanpercentile(self.replaced,16)
		vmax=np.nanpercentile(self.replaced,90)
		plt.figure(figsize=(8,4))
		plt.subplot(131)
		plt.title('PS1 image')
		plt.imshow(self.ps1.padded,vmax=vmax,vmin=vmin)
		plt.plot(self.ps1satcat['x'].values,self.ps1satcat['y'].values,'C1+')
		plt.subplot(132)
		plt.title('Injected sources')
		plt.imshow(self.inject,vmax=vmax,vmin=vmin)
		plt.plot(self.ps1satcat['x'].values,self.ps1satcat['y'].values,'C1+')
		plt.subplot(133)
		plt.title('Source replaced')
		plt.imshow(self.replaced,vmax=vmax,vmin=vmin)
		plt.plot(self.ps1satcat['x'].values,self.ps1satcat['y'].values,'C1+')
		plt.tight_layout()


## Tools


