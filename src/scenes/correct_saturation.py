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

#from .tools import query_ps1, ps_psf, psf_minimizer, psf_phot, mask_rad_func


def image2counts(data,header,toflux=True):
	if toflux:
		a = 2.5/np.log(10)
		x = data/a
		flux = header['boffset'] + header['bsoften'] * 2 * np.sinh(x)
		return flux
	else:
		a = 2.5/np.log(10)
		tmp = (data - header['boffset']) / (header['bsoften']*2)
		tmp = np.arcsinh(tmp)*a
		return tmp

class saturated_stars():
	def __init__(self,file,savepath,catalogpath=None,mask_file=None,
				 satlim=14,calmags=[15,17],run=True,overwrite=False):
		"""
		"""
		self.file = file 
		self.mask_file = mask_file
		self._load_file()
		self.savepath = savepath
		self.overwrite = overwrite
		self.satlim = satlim
		self.calmags = calmags
		self._get_catalog(catalogpath)

		if run:
			self.fit_psf()
			self.flux_offset()
			self.replace_saturation()
			self._save_image()
			if self.mask is not None:
				self._save_mask()



	def _load_file(self):
		hdu = fits.open(self.file)
		self.header = hdu[0].header

		self.data = image2counts(hdu[0].data,self.header)
		self.wcs = WCS(self.header)
		self.ra, self.dec = self.wcs.all_pix2world(self.data.shape[1]/2,self.data.shape[0]/2,1)
		self.image_stats = sigma_clipped_stats(self.data)
		self.band = self.file.split('stk.')[-1].split('.unconv')[0]
		self.zp = 25 + 2.5*np.log10(hdu[0].header['EXPTIME'])

		hdu.close()
		if self.mask_file is not None:
			hdu = fits.open(self.mask_file)
			self.mask_header = hdu[0].header
			self.mask = hdu[0].data
			hdu.close()
		else:
			self.mask = None
			self.mask_header = None


	def _get_catalog(self,catalogpath=None):
		if catalogpath is None:
			cat = query_ps1(self.ra,self.dec,0.4)
		else:
			cat = pd.read_csv(catalogpath)

		x,y = self.wcs.all_world2pix(cat.raMean.values,cat.decMean.values,0)
		cat['x'] = x; cat['y'] = y
		ind = (x > 5) & (x < self.data.shape[1]-5) & (y > 5) & (y < self.data.shape[0]-5)
		cat = cat.iloc[ind]
		cat = cat.loc[(cat['iMeanPSFMag'] > 0) & (cat['rMeanPSFMag'] > 0) & 
					  (cat['zMeanPSFMag'] > 0) & (cat['yMeanPSFMag'] > 0)]
		cat = cat.sort_values(self.band+'MeanPSFMag')
		self.ps1cat = cat
		self.ps1satcat = deepcopy(cat.loc[cat[self.band+'MeanPSFMag'] < self.satlim])
		self.calcat = self.ps1cat.loc[(self.ps1cat[self.band+'MeanPSFMag'] > self.calmags[0]) & (self.ps1cat[self.band+'MeanPSFMag'] < self.calmags[1])]



	def fit_psf(self,size=15):
		cal = self.calcat
		psf_mod = []
		for j in range(len(cal)):
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
			cflux = 10**((cal[self.band+'MeanPSFMag'].values[j]-self.zp)/-2.5)
			f = minimize(psf_phot,cflux,args=(cut,psf))
			fit_flux += [f.x]
		fit_flux = np.array(fit_flux)
		factor = np.nanmedian(fit_flux / 10**((cal[self.band+'MeanPSFMag'].values-self.zp)/-2.5))
		self.flux_factor = factor
		self._fitflux = fit_flux

	def _create_cut(self,source,size):
		xx = source['x']; yy = source['y']
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
			psf = ps_psf(self.psf_param[:-2],x-self.psf_param[-2]-rad,y-self.psf_param[-1]-rad)
			
			xx = sat['x'].values[i]; yy = sat['y'].values[i]
			dist = np.sqrt(((x-rad))**2 + ((y-rad))**2)
			ind = np.array(np.where(dist < rad))
			ind[0] += int(yy) - rad
			ind[1] += int(xx) - rad
			good = (ind[0] >= 0) & (ind[1] >= 0) & (ind[0] < self.data.shape[0]) & (ind[1] < self.data.shape[1])
			ind = ind[:,good]
			masking[ind[0],ind[1]] = np.nan
			dx = min(ind[1])
			dy = min(ind[0])
			
			inject[ind[0],ind[1]] += psf[ind[0]-dy,ind[1]-dx] * cflux*self.flux_factor + self.image_stats[1]
		if self.mask is not None:
			masking[self.mask>0] = self.image_stats[1] # set to the median
			self.newmask = inject > 0
		replaced = np.nansum([masking,inject],axis=0)
		self.replaced = replaced
		self.inject = inject


	def _save_image(self):
		newheader = deepcopy(self.header)
		newheader['SATCOR'] = (True,'Corrected sat stars')
		newheader['SATDATE'] = (date.today().isoformat(),'Date of correction')
		data = image2counts(self.replaced,self.header,toflux=False)
		hdu = fits.PrimaryHDU(data=data,header=newheader)
		hdul = fits.HDUList([hdu])

		savename = self.savepath + self.file.split('fits')[0].split('/')[-1] + 'satcor.fits'
		hdul.writeto(savename,overwrite=self.overwrite)

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
		savename = self.savepath + self.mask_file.split('fits')[0].split('/')[-1] + 'satcor.fits'
		hdul.writeto(savename,overwrite=self.overwrite)

	def plot_catvspsf(self):
		plt.figure()
		plt.plot(self.calcat[self.band+'MeanPSFMag'],self._fitflux,label='PSF flux')
		plt.plot(self.calcat[self.band+'MeanPSFMag'],10**((self.calcat[self.band+'MeanPSFMag'].values-self.zp)/-2.5)*self.flux_factor,label='Corrected catalog flux')
		plt.xlabel(self.band+' mag')
		plt.ylabel('Counts')
		plt.legend()
		plt.title('Correction: ' + str(np.round(self.flux_factor,2))+r'$\times$flux')

	def plot_result(self):
		vmin=np.nanpercentile(self.replaced,16)
		vmax=np.nanpercentile(self.replaced,90)
		plt.figure(figsize=(8,4))
		plt.subplot(131)
		plt.title('PS1 image')
		plt.imshow(self.data,vmax=vmax,vmin=vmin)
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


import pandas as pd
import numpy as np
from scipy.optimize import minimize 


def isolate_stars(cat,only_stars=False,Qf_lim=0.85,psfkron_diff=0.05):
    qf_ind = ((cat.gQfPerfect.values > Qf_lim) & (cat.rQfPerfect.values > Qf_lim) & 
              (cat.iQfPerfect.values > Qf_lim) & (cat.zQfPerfect.values > Qf_lim))
    kron_ind = (cat.rMeanPSFMag.values - cat.rMeanKronMag.values) < psfkron_diff
    ind = qf_ind & kron_ind
    if only_stars:
        cat = cat.iloc[ind]
        cat.loc[:,'star'] = 1
    else:
        cat.loc[:,'star'] = 0
        cat.loc[ind,'star'] = 1
    return cat 

def cut_bad_detections(cat):
    ind = (cat.rMeanPSFMag.values > 0) & (cat.iMeanPSFMag.values > 0) & (cat.zMeanPSFMag.values > 0)
    return cat.iloc[ind]


def query_ps1(ra,dec,radius,only_stars=False,version='dr2'):
    if (version.lower() != 'dr2') & (version.lower() != 'dr1'):
        m = 'Version must be dr2, or dr1'
        raise ValueError(m)
    
    str = f'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/{version.lower()}/mean?ra={ra}&dec={dec}&radius={radius}&nDetections.gte=5&pagesize=-1&format=csv'
    try:
        cat = pd.read_csv(str)
    except pd.errors.EmptyDataError:
        print('No detections')
        cat = []
    cat = isolate_stars(cat,only_stars=only_stars)
    cat = cut_bad_detections(cat)
    return cat 


def ps_psf(params,x,y):
    sdx,sdy,sdxy,k = params
    z = x**2/(2*sdx**2) + y**2/(2*sdy**2) + sdxy*x*y
    #psf = (1 + k*z + z*2.25)**(-1)
    psf = (1 + z + z**2/2 + z**3/6)**(-1) # PGAUSS
    psf /= np.nansum(psf)
    return psf

def psf_minimizer(x0,cut,x,y):
    c = deepcopy(cut)
    #y, x = np.mgrid[:c.shape[0], :c.shape[1]]
    #x = x - c.shape[1] / 2 #+ x0[-2]
    #y = y - c.shape[0] / 2 #+ x0[-1]
    x = x - x0[-2]
    y = y - x0[-1]
    psf = ps_psf(x0[:-2],x,y)
    c /= np.nansum(c)
    res = np.nansum((c-psf)**2)
    return res


def psf_phot(f,cut,psf):
    res = np.nansum((cut-psf*f)**2)
    return res

def mask_rad_func(x, a=7.47132813e+03, b=5.14848986e-01, c=3.15022393e+01):
    rad = np.array(a * np.exp(-b * x) + c)
    rad[rad>300] = 300
    rad = rad.astype(int)
    return rad