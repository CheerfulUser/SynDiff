from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
import numpy as np
import pandas as pd
from datetime import date

from scipy.optimize import minimize 
from copy import deepcopy
from astropy.stats import sigma_clipped_stats

from .tools import query_ps1, ps_psf, psf_minimizer, psf_phot, mask_rad_func


def image2counts(data,header,toflux=True):
	if toflux:
		a = 2.5/np.log(10)
		x = data/a
		flux = header['boffset'] + header['bsoften'] * 2 * np.sinh(x)
		return flux
	else:
		tmp = (data - header['boffset']) / (header['bsoften']*2)
		tmp = np.arcsinh(tmp)*a
		return tmp

class saturated_stars():
	def __init__(self,file,savepath,catalogpath=None,mask_file=None,satlim=14,calmags=[15,17],run=True):
		"""
		"""

		self.file = file 
		self.mask_file = mask_file
		self._load_file()

		self.satlim = satlim
		self.calmags = calmags
		self._get_catalog(catalogpath)

		if run:
			self.fit_psf()
			self.flux_offset()
			self.replace_saturation()
			self._save_image()
			self._save_mask()



	def _load_file(self):
		hdu = fits.open(file)
		self.header = hdu[0].header

		self.data = image2counts(fits.open(files[fnum])[0].data,self.header)
		self.wcs = WCS(self.header)
		self.ra, self.dec = wcs.all_pix2world(self.data.shape[1]/2,self.data.shape[0]/2,1)
		self.image_stats = sigma_clipped_stats(self.data)
		self.band = self.file.split('stk.')[-1].split('.unconv')[0]
		self.zp = 25 + 2.5*np.log10(hdu[0].header['EXPTIME'])

		hdu.close()
		if self.mask_file is not None:
			hdu = fits.open(file)
			self.mask_header = hdu[0].header
			self.mask = hdu[0].data
			hdu.close()


	def _get_catalog(self,catalogpath=None):
		if catalogpath is None:
			cat = query_ps1(self.ra,self.dec,0.4)
		else:
			cat = pd.read_csv(catalogpath)

		x,y = wcs.all_world2pix(cat.raMean.values,cat.decMean.values,0)
		cat['x'] = x; cat['y'] = y
		ind = (x > 5) & (x < self.data.shape[1]-5) & (y > 5) & (y < self.data.shape[0]-5)
		cat = cat.iloc[ind]
		cat = cat.loc[(cat['iMeanPSFMag'] > 0) & (cat['rMeanPSFMag'] > 0) & 
					  (cat['zMeanPSFMag'] > 0) & (cat['yMeanPSFMag'] > 0)]
		cat = cat.sort_values(self.band+'MeanPSFMag')
		self.ps1cat = cat
		self.ps1satcat = deepcopy(cat.loc[cat[self.band+'MeanPSFMag'] < self.satlim])
		self.calcat = self.ps1cat.loc[(self.ps1cat[self.band+'MeanPSFMag'] > 15) & (self.ps1cat[self.band+'MeanPSFMag'] < 16)]



	def fit_psf(self,size=15):
		cal = self.calcat
		psf_mod = []
		for j in range(len(cal)):
			cut, x, y = self._create_cut(cal.iloc[j])
			x0 = [10,10,0,1,0,0]
			res = minimize(psf_minimizer,x0,args=(cut,x,y))
			psf_mod += [res.x]
		psf_mod = np.array(psf_mod)
		psf_param = np.nanmedian(psf_mod,axis=0)
		self.psf_param = psf_param

	def flux_offset(self,size=15):
		cal = tab.loc[(tab[band+'MeanPSFMag'] > 15) & (tab[band+'MeanPSFMag'] < 17)]
		fit_flux = []
		for j in range(len(cal)):
			cut, x, y = self._create_cut(cal.iloc[j])
			psf = PS_PSF(self.psf_param[:-2],x-self.psf_param[-2],y-self.psf_param[-1])
			cflux = 10**((cal[self.band+'MeanPSFMag'].values[j]-zp)/-2.5)
			f = minimize(PSF_phot,cflux,args=(cut,psf))
			fit_flux += [f.x]
		fit_flux = np.array(fit_flux)
		factor = np.nanmedian(fit_flux / 10**((cal[self.band+'MeanPSFMag'].values-self.zp)/-2.5))
		self.flux_factor = factor

	def _create_cut(self,source,size):
		xx = source['x'].values; yy = source['y'].values
		yint = int(yy+0.5)
		xint = int(xx+0.5)
		cut = deepcopy(self.data[yint-size:yint+size+1,xint-size:xint+size+1])
		y, x = np.mgrid[:cut.shape[0], :cut.shape[1]]
		x = x - cut.shape[1] / 2 - (xx-xint)
		y = y - cut.shape[0] / 2 - (yy-yint)
		return cut,x,y


	def replace_saturation(self):
		masking = deepcopy(self.data)
		inject = np.zeros_like(self.data)
		sat = self.ps1satcat
		rads = mask_rad_func(sat[self.band+'MeanPSFMag'].values)
		cfluxes = 10**((sat[self.band+'MeanPSFMag'].values-self.zp)/-2.5)
		for i in range(len(sat)):
			rad = rads[i]
			cflux = cfluxes[i]
			y,x = np.mgrid[:rad*2,:rad*2]
			psf = ps_psf(psf_param[:-2],x-psf_param[-2]-rad,y-psf_param[-1]-rad)
			
			xx = sat['x'].values[i]; yy = sat['y'].values[i]
			dist = np.sqrt(((x-rad))**2 + ((y-rad))**2)
			ind = np.array(np.where(dist < rad))
			ind[0] += int(yy) - rad
			ind[1] += int(xx) - rad
			good = (ind[0] >= 0) & (ind[1] >= 0) & (ind[0] < flux.shape[0]) & (ind[1] < flux.shape[1])
			ind = ind[:,good]
			masking[ind[0],ind[1]] = np.nan
			tmask = time.time()
			dx = min(ind[1])
			dy = min(ind[0])
			
			inject[ind[0],ind[1]] += psf[ind[0]-dy,ind[1]-dx] * cflux*factor + med
		if self.mask is not None:
			masking[self.mask>0] = self.image_stats[1] # set to the median
			self.newmask = inject > 0
		replaced = np.nansum([masking,inject],axis=0)
		self.replaced = replaced


	def _save_image(self):
		newheader = deepcopy(self.header)
		newheader['SATCOR'] = (True,'Corrected sat stars')
		newheader['SATDATE'] = (date.today().isoformat(),'Date of correction')
		data = image2counts(self.replaced,toflux=False)
		hdu = fits.PrimaryHDU(data=data,header=newheader)
		hdul = fits.HDUList([hdu])

		savename = savepath + self.file.split('fits')[0].split('/')[-1] + 'satcor.fits'
		hdul.writeto(savename)

	def _save_mask(self):
		newheader = deepcopy(self.mask_header)
		newheader['SATCOR'] = (True,'Corrected sat stars')
		newheader['SATBIT'] = (0x8000,'Bit for the saturation mask')
		newheader['SATDATE'] = (date.today().isoformat(),'Date of correction')
		m = deepcopy(self.newmask).astype(int)
		m[m > 0] = 0x8000
		data = self.mask | m
		hdu = fits.PrimaryHDU(data=data,header=newheader)
		hdul = fits.HDUList([hdu])
		savename = savepath + self.file.split('fits')[0].split('/')[-1] + 'satcor.fits'
		hdul.writeto(savename)





