from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

from scipy.optimize import minimize 
from scipy.signal import fftconvolve
from copy import deepcopy
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from photutils.aperture import CircularAperture
from photutils.aperture import ApertureStats, aperture_photometry
# from astroquery.vizier import Vizier
# Vizier.ROW_LIMIT = -1

from tools import * #ps_psf, psf_minimizer, psf_phot, mask_rad_func
from tools import _get_gaia, _get_bsc
from timeit import default_timer as timer



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

def new_get_catalog(catalog, wcs, padded, band, pad = 500, ra = None, dec = None):
	if (type(catalog) == bool):
		cat = query_ps1(ra, dec,0.41)
	elif catalog is not None:
		if type(catalog) == str:
			cat = pd.read_csv(catalog)
		else:
			cat = catalog
	else:
		return 
	
	x,y = wcs.all_world2pix(cat['raMean'].values,cat['decMean'].values,0)
	x += pad; y += pad
	cat['x'] = x
	cat['y'] = y
	ind = (x > 0) & (y > 0) & (x < padded.shape[1]) & (y < padded.shape[0])
	cat = cat.iloc[ind]
	cat = cat.loc[(cat['iMeanPSFMag'] > 0) & (cat['rMeanPSFMag'] > 0) & 
					(cat['zMeanPSFMag'] > 0) & (cat['yMeanPSFMag'] > 0)]
	cat = cat.sort_values(band+'MeanPSFMag')
	cat = cat.drop_duplicates()
	return cat

class saturated_stars():
	def __init__(self,ps1,catalog_ps1=None,catalogpath=None,mask=True,
				 satlim=None,calmags=[15,17],run=True,overwrite=False):
		"""
		"""
		start = timer()
		self.ps1 = ps1 
		self.band = ps1.band
		self.wcs = ps1.wcs
		# self._load_ps1(mask)
		self.padded_skycell = ps1.padded
		#self.savepath = savepath
		self.overwrite = overwrite
		#if self._check_exist():
		self._set_satlim(satlim)
		self.calmags = calmags
		prelim = timer() - start
		#print('prelim ',prelim)
		self.sort_cats(catalog_ps1,catalogpath)
		#cats = timer() - prelim
		#print('cats ',cats)
		if run:
			self.fit_psf()
			#psf = timer() - cats
			#print('psf ',cats)
			self.flux_offset()
			#offset = timer() - psf
			#print('offset ',offset)
			self.kill_saturation()
			self.replace_saturation()
			#replace = timer() - offset
			#print('replace ',replace)
			self._update_image()
			#uimage = timer() - replace
			#print('update image ',uimage)
			self._update_mask()
			#umask = timer() - uimage
			#print('update mask ',umask)
				#if self.ps1.mask is not None:
				#	self._save_mask()



	def _check_exist(self):
		savename = self.savepath + self.file.split('fits')[0].split('/')[-1] + 'satcor.fits'
		run = True
		try:
			hdu = fits.open(savename)
			hdu.close()
			if ~self.overwrite:
				run = False
				print('File exists')
			else:
				print('Overwriting file')
		except:
			pass
		return run

	def _load_ps1(self,mask):
		if type(self.ps1) == str:
			self.ps1 = ps1_data(self.ps1,mask=mask)
		# self.ps1.convert_flux_scale(toflux=True)

	def _set_satlim(self,satlim):
		if satlim is None:
			offset = 0
			given = {'g':14.5-offset,'r':15-offset,'i':15-offset,'z':14-offset,'y':13-offset}
			self.satlim = given[self.band]
		else:
			self.satlim = satlim


	def _get_catalog(self,catalog_ps1=None):
		
		# self.ps1cat = pd.read_csv(catalog_ps1)
     
		# self.ps1.get_catalog(catalog_ps1)
		# cat = new_get_catalog(catalog_ps1, wcs, pad, self.padded_skycell)
		cat = new_get_catalog(catalog_ps1, self.wcs, self.padded_skycell, self.band, pad = 500, ra = None, dec = None)
  
		if len(cat) < 1:
			# print(cat)
			print('ZZZ No catalog found, using PS1 catalog')
			print('ZZZ catalog_ps1:', catalog_ps1, type(catalog_ps1))
  
		# print('PS1 catalog loaded with',len(self.ps1.cat),'sources', type(self.ps1.cat))

		self.ps1cat = deepcopy(cat)
		self.ps1satcat = deepcopy(self.ps1cat.loc[self.ps1cat[self.band+'MeanPSFMag'] < self.satlim])
		self.calcat = deepcopy(self.ps1cat.loc[(self.ps1cat[self.band+'MeanPSFMag'] > self.calmags[0]) & (self.ps1cat[self.band+'MeanPSFMag'] < self.calmags[1])])

	def _get_vizier_cats(self,catalogpath=None):
		if catalogpath is None:
			sc = SkyCoord(self.ps1.ra,self.ps1.dec, frame='icrs', unit='deg')
			gaia = _get_gaia(sc,0.41)
			bsc = _get_bsc(sc,0.41)
			#tyco = _get_tyco(sc,0.8)
		else:
			try:
				gaia = pd.read_csv(catalogpath+'_gaia.csv')
			except:
				gaia = None
    
			try:
   				bsc = pd.read_csv(catalogpath+'_bsc.csv')
			except:
				bsc = None
			#tyco = pd.read_csv(catalogpath+'tyco.csv')
		if gaia is not None:
			x,y = self.ps1.wcs.all_world2pix(gaia.RA_ICRS.values,gaia.DE_ICRS.values,0)
			x += self.ps1.pad; y += self.ps1.pad
			gaia['x'] = x; gaia['y'] = y
			ind = (x > 5) & (x < self.padded_skycell.shape[1]-5) & (y > 5) & (y < self.padded_skycell.shape[0]-5) # should change this to oversize
			self.gaiacat = gaia.iloc[ind]
		else:
			self.gaiacat = None
		if bsc is not None:
			x,y = self.ps1.wcs.all_world2pix(bsc.RA_ICRS.values,bsc.DE_ICRS.values,0)
			x += self.ps1.pad; y += self.ps1.pad
			bsc['x'] = x; bsc['y'] = y
			ind = (x > 5) & (x < self.padded_skycell.shape[1]-5) & (y > 5) & (y < self.padded_skycell.shape[0]-5) # should change this to oversize
			self.bsccat = bsc.iloc[ind]
		else:
			self.bsccat = None

		#if tyco is not None:
		#	x,y = self.ps1.wcs.all_world2pix(tyco.RA_ICRS.values,tyco.DE_ICRS.values,0)
		#	tyco['x'] = x; tyco['y'] = y
		#	ind = (x > 5) & (x < self.padded_skycell.shape[1]-5) & (y > 5) & (y < self.padded_skycell.shape[0]-5) # should change this to oversize
		#	self.tycocat = tyco.iloc[ind]
		#else:
		#	self.tycocat = None


	def _get_atlas_refcat(self,replace_ps1=True):
		cat = search_refcat(self.ps1.ra,self.ps1.dec,0.4)
		if replace_ps1:
			offset = 0
			given = {'g':14.5-offset,'r':15-offset,'i':15-offset,'z':14-offset,'y':13-offset}
			bands = ['r','i','z']
			for ind in cat['objid'].values:
				row = cat.loc[cat['objid'] == ind]
				if len(self.ps1cat.loc[self.ps1cat['objID']==ind]) > 0:
					for band in bands:
						self.ps1cat[band+'PSFMeanMag'].loc[self.ps1cat['objID']==ind] = row[band]
				else:
					r = deepcopy(self.ps1cat)

				
	def _kill_cat(self):
		kill_cat = None
		if self.gaiacat is not None:
			ugaia = find_unique2ps1(self.ps1cat,self.gaiacat, self.ps1.temp_catalog)
   
			if ugaia is None:
				print(self.ps1cat)

			kill_cat = ugaia[['x','y','Gmag']]
			kill_cat = kill_cat.rename(columns={'Gmag':'mag'})
		if self.bsccat is not None:
			ubsc = find_unique2viz(self.gaiacat,self.bsccat)
			ubsc = ubsc[['x','y','Vmag']]
			ubsc = ubsc.rename(columns={'Vmag':'mag'})
			kill_cat = pd.concat([kill_cat,ubsc])

		if kill_cat is not None:
			kill_cat = combine_close_sources(kill_cat)
		self.kill_cat = kill_cat


	def sort_cats(self,catalog_ps1,catalogpath):
		self._get_catalog(catalog_ps1)
		self._get_vizier_cats(catalogpath)
		self._kill_cat()
		



	def fit_psf(self,size=15):
		cal = self.calcat
		psf_mod = []
		if len(cal) > 30:
			r = 30
		else:
			r = len(cal)
		for j in range(r):
			cut, x, y = self._create_cut(cal.iloc[j],size)
			x0 = [10,10,0,1,0,0]
			res = minimize(psf_minimizer,x0,args=(cut,x,y))
			psf_mod += [res.x]
		psf_mod = np.array(psf_mod)
		psf_param = np.nanmedian(psf_mod,axis=0)
		self.psf_param = psf_param

	def flux_offset(self,size=15,psf=False):
		cal = self.calcat
		if psf:
			fit_flux = []
			for j in range(len(cal)):
				cut, x, y = self._create_cut(cal.iloc[j],size)
				psf = ps_psf(self.psf_param[:-2],x-self.psf_param[-2],y-self.psf_param[-1])
				cflux = 10**((cal[self.band+'MeanPSFMag'].values[j]-25)/-2.5)
				f = minimize(psf_phot,cflux,args=(cut,psf))
				fit_flux += [f.x]
			fit_flux = np.array(fit_flux)
		else:
			pos = list(zip(cal.x.values, cal.y.values))
			aperture = CircularAperture(pos, 20)
			m,med,std = sigma_clipped_stats(self.padded_skycell)
			origphot = aperture_photometry(self.padded_skycell-med, aperture)
			fit_flux = origphot['aperture_sum'].value
		factor = np.nanmedian(fit_flux / 10**((cal[self.band+'MeanPSFMag'].values-25)/-2.5))
		self.flux_factor = factor
		self._fitflux = fit_flux

	def _create_cut(self,source,size):
		xx = source['x']; yy = source['y']
		yint = int(yy+0.5)
		xint = int(xx+0.5)
		cut = deepcopy(self.padded_skycell[yint-size:yint+size+1,xint-size:xint+size+1])
		y, x = np.mgrid[:cut.shape[0], :cut.shape[1]]
		x = x - cut.shape[1] / 2 - (xx-xint)
		y = y - cut.shape[0] / 2 - (yy-yint)
		return cut,x,y

	def kill_saturation(self):
		masking = deepcopy(self.padded_skycell)
		if self.ps1.mask is not None:
			masking[self.ps1.mask>0] = self.ps1.image_stats[1]
			masking[np.isnan(masking)] = self.ps1.image_stats[1]
			self.newmask = np.isnan(masking)
		if self.kill_cat is not None:
			sat = self.kill_cat
			rads = mask_rad_func(sat['mag'].values)
			for i in range(len(sat)):
				rad = rads[i]
				y,x = np.mgrid[:rad*2,:rad*2]
				
				xx = sat['x'].values[i]; yy = sat['y'].values[i]
				dist = np.sqrt(((x-rad))**2 + ((y-rad))**2)
				ind = np.array(np.where(dist < rad))
				ind[0] += int(yy) - rad
				ind[1] += int(xx) - rad
				good = (ind[0] >= 0) & (ind[1] >= 0) & (ind[0] < self.padded_skycell.shape[0]) & (ind[1] < self.padded_skycell.shape[1])
				ind = ind[:,good]
				masking[ind[0],ind[1]] = np.nan
		self.masking = masking
			


	def replace_saturation(self):
		masking = deepcopy(self.masking)
		inject = np.zeros_like(self.padded_skycell)
		sat = self.ps1satcat
		live = np.isfinite(masking[sat['y'].values.astype(int),sat['x'].values.astype(int)])
		sat = sat.iloc[live]
		rads = mask_rad_func(sat[self.band+'MeanPSFMag'].values)
		r2 = (rads * 2.5).astype(int)
		suspect = np.zeros_like(masking,dtype=int)
		cfluxes = 10**((sat[self.ps1.band+'MeanPSFMag'].values-25)/-2.5)
		for i in range(len(sat)):
			xx = sat['x'].values[i]; yy = sat['y'].values[i]
			#val = killed[int(yy),int(xx)]
			#if np.isfinite(val):
			rad = rads[i]
			cflux = cfluxes[i]
			y,x = np.mgrid[:rad*2,:rad*2]
			psf = ps_psf(self.psf_param[:-2],x-self.psf_param[-2]-rad,y-self.psf_param[-1]-rad)
			
			xx = sat['x'].values[i]; yy = sat['y'].values[i]
			dist = np.sqrt(((x-rad))**2 + ((y-rad))**2)
			ind = np.array(np.where(dist < rad))
			ind[0] += int(yy) - rad
			ind[1] += int(xx) - rad
			#print(np.nansum(psf[ind[0]-min(ind[0]),ind[1]-min(ind[1])]))
			#psf /= np.nansum(psf[ind[0]-min(ind[0]),ind[1]-min(ind[1])])
			dx = min(ind[1])
			dy = min(ind[0])
			good = (ind[0] >= 0) & (ind[1] >= 0) & (ind[0] < self.padded_skycell.shape[0]) & (ind[1] < self.padded_skycell.shape[1])
			ind = ind[:,good]
			masking[ind[0],ind[1]] = np.nan
			inject[ind[0],ind[1]] += psf[ind[0]-dy,ind[1]-dx] * cflux * self.flux_factor + self.ps1.image_stats[1]

			y,x = np.mgrid[:r2[i]*2,:r2[i]*2]
			dist = np.sqrt(((x-r2[i]))**2 + ((y-r2[i]))**2)
			ind = np.array(np.where(dist < r2[i]))
			ind[0] += int(yy) - r2[i]
			ind[1] += int(xx) - r2[i]
			good = (ind[0] >= 0) & (ind[1] >= 0) & (ind[0] < self.padded_skycell.shape[0]) & (ind[1] < self.padded_skycell.shape[1])
			ind = ind[:,good]
			suspect[ind[0],ind[1]] = 0x0080
			#inject[ind[0],ind[1]] += psf[ind[0]-dy,ind[1]-dx] + self.ps1.image_stats[1]
		if self.ps1.mask is not None:
			#masking[self.ps1.mask>0] = self.ps1.image_stats[1] # set to the median
			self.newmask = (self.newmask + (inject > 0)) > 0
			self.suspect = suspect
		replaced = np.nansum([masking,inject],axis=0)
		replaced[replaced == 0] = self.ps1.image_stats[1]
		self.replaced = replaced
		self.inject = inject

	def _update_image(self):
		newheader = deepcopy(self.ps1.header)
		newheader['SATCOR'] = (True,'Corrected sat stars')
		newheader['SATDATE'] = (date.today().isoformat(),'Date of correction')
		self.ps1.header = newheader
		self.padded_skycell = self.replaced

	def _update_mask(self):
		if self.ps1.mask is not None:
			newheader = deepcopy(self.ps1.mask_header)
			newheader['SATCOR'] = (True,'Corrected sat stars')
			newheader['SATDATE'] = (date.today().isoformat(),'Date of correction')
			m = deepcopy(self.newmask).astype(int)
			m[m > 0] = 0x0020

			self.ps1.mask = self.ps1.mask.astype(int)  # or np.uint32, depending on the bit size needed
			m = m.astype(int)  # or np.uint32
			# print('ZZZ Masking shape', m.shape, type(m))
			data = self.ps1.mask | m
			#self.suspect[self.suspect] = 0x0080
			data = data | self.suspect
			self.ps1.mask = data
			self.ps1.mask_header = newheader

	def plot_flux_correction(self):
		plt.figure()
		plt.semilogy(self.calcat[self.band+'MeanPSFMag'],self._fitflux,'.',label='Image flux')
		plt.plot(self.calcat[self.band+'MeanPSFMag'],10**((self.calcat[self.band+'MeanPSFMag'].values-25)/-2.5)*self.flux_factor,label='Corrected catalog flux')
		plt.xlabel(self.band+' mag',fontsize=15)
		plt.ylabel('Counts',fontsize=15)
		plt.legend()
		plt.title('Correction: ' + str(np.round(self.flux_factor,2))+rf'$f_{self.band}$')

	def plot_result(self):
		vmin=np.nanpercentile(self.replaced,16)
		vmax=np.nanpercentile(self.replaced,90)
		plt.figure(figsize=(8,4))
		plt.subplot(132)
		plt.title('PS1 image')
		plt.imshow(self.padded_skycell,vmax=vmax,vmin=vmin,origin='lower')
		plt.plot(self.ps1satcat['x'].values,self.ps1satcat['y'].values,'C1+')
		if self.kill_cat is not None:
			plt.plot(self.kill_cat['x'].values,self.kill_cat['y'].values,'C3x')
		plt.subplot(131)
		plt.title('Injected sources')
		plt.imshow(self.inject,vmax=vmax,vmin=vmin,origin='lower')
		plt.plot(self.ps1satcat['x'].values,self.ps1satcat['y'].values,'C1+',label='PS1 catalog')
		if self.kill_cat is not None:
			plt.plot(self.kill_cat['x'].values,self.kill_cat['y'].values,'C3x',label='Gaia + BSC')
		plt.legend(loc='lower left')
		plt.subplot(133)
		plt.title('Source replaced')
		plt.imshow(self.replaced,vmax=vmax,vmin=vmin,origin='lower')
		plt.plot(self.ps1satcat['x'].values,self.ps1satcat['y'].values,'C1+',label='PS1 catalog')
		if self.kill_cat is not None:
			plt.plot(self.kill_cat['x'].values,self.kill_cat['y'].values,'C3x',label='Gaia + BSC')
		plt.tight_layout()
