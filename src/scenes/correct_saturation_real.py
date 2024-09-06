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
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from photutils.aperture import CircularAperture
from photutils.aperture import ApertureStats, aperture_photometry
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

from tools import ps_psf, psf_minimizer, psf_phot, mask_rad_func


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
				 satlim=None,calmags=[15,17],oversize=False,run=True,overwrite=False):
		"""
		"""
		self.file = file 
		self.mask_file = mask_file
		self._load_file()
		self.savepath = savepath
		self.overwrite = overwrite
		if self._check_exist():
			self._set_satlim(satlim)
			self.calmags = calmags
			self.sort_cats(catalogpath)
			if run:
				self.fit_psf()
				self.flux_offset()
				self.kill_saturation()
				self.replace_saturation()
				self._save_image()
				#if self.mask is not None:
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

	def _set_satlim(self,satlim):
		if satlim is None:
			offset = 0
			given = {'g':14.5-offset,'r':15-offset,'i':15-offset,'z':14-offset,'y':13-offset}
			self.satlim = given[self.band]
		else:
			self.satlim = satlim


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

	def _get_vizier_cats(self,catalogpath=None):
		if catalogpath is None:
			sc = SkyCoord(self.ra,self.dec, frame='icrs', unit='deg')
			gaia = _get_gaia(sc,0.41)
			bsc = _get_bsc(sc,0.41)
			#tyco = _get_tyco(sc,0.8)
		else:
			gaia = pd.read_csv(catalogpath+'gaia.csv')
			bsc = pd.read_csv(catalogpath+'bsc.csv')
			tyco = pd.read_csv(catalogpath+'tyco.csv')
		if gaia is not None:
			x,y = self.wcs.all_world2pix(gaia.RA_ICRS.values,gaia.DE_ICRS.values,0)
			gaia['x'] = x; gaia['y'] = y
			ind = (x > 5) & (x < self.data.shape[1]-5) & (y > 5) & (y < self.data.shape[0]-5) # should change this to oversize
			self.gaiacat = gaia.iloc[ind]
		else:
			self.gaiacat = None
		if bsc is not None:
			x,y = self.wcs.all_world2pix(bsc.RA_ICRS.values,bsc.DE_ICRS.values,0)
			bsc['x'] = x; bsc['y'] = y
			ind = (x > 5) & (x < self.data.shape[1]-5) & (y > 5) & (y < self.data.shape[0]-5) # should change this to oversize
			self.bsccat = bsc.iloc[ind]
		else:
			self.bsccat = None

		#if tyco is not None:
		#	x,y = self.wcs.all_world2pix(tyco.RA_ICRS.values,tyco.DE_ICRS.values,0)
		#	tyco['x'] = x; tyco['y'] = y
		#	ind = (x > 5) & (x < self.data.shape[1]-5) & (y > 5) & (y < self.data.shape[0]-5) # should change this to oversize
		#	self.tycocat = tyco.iloc[ind]
		#else:
		#	self.tycocat = None


	def _get_atlas_refcat(self,replace_ps1=True):
		cat = search_refcat(self.ra,self.dec,0.4)
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
			ugaia = find_unique2ps1(self.ps1cat,self.gaiacat)
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


	def sort_cats(self,catalogpath):
		self._get_catalog(catalogpath)
		self._get_vizier_cats(catalogpath)
		self._kill_cat()
		



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

	def flux_offset(self,size=15,psf=False):
		cal = self.calcat
		if psf:
			fit_flux = []
			for j in range(len(cal)):
				cut, x, y = self._create_cut(cal.iloc[j],size)
				psf = ps_psf(self.psf_param[:-2],x-self.psf_param[-2],y-self.psf_param[-1])
				cflux = 10**((cal[self.band+'MeanPSFMag'].values[j]-self.zp)/-2.5)
				f = minimize(psf_phot,cflux,args=(cut,psf))
				fit_flux += [f.x]
			fit_flux = np.array(fit_flux)
		else:
			pos = list(zip(cal.x.values, cal.y.values))
			aperture = CircularAperture(pos, 20)
			m,med,std = sigma_clipped_stats(self.data)
			origphot = aperture_photometry(self.data-med, aperture)
			fit_flux = origphot['aperture_sum'].value
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

	def kill_saturation(self):
		masking = deepcopy(self.data)
		if self.mask is not None:
			masking[self.mask>0] = self.image_stats[1]
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
				good = (ind[0] >= 0) & (ind[1] >= 0) & (ind[0] < self.data.shape[0]) & (ind[1] < self.data.shape[1])
				ind = ind[:,good]
				masking[ind[0],ind[1]] = np.nan
		self.masking = masking
		if self.mask is not None:
			self.newmask = np.isnan(masking)


	def replace_saturation(self):
		masking = deepcopy(self.masking)
		inject = np.zeros_like(self.data)
		sat = self.ps1satcat
		rads = mask_rad_func(sat[self.band+'MeanPSFMag'].values)
		cfluxes = 10**((sat[self.band+'MeanPSFMag'].values-self.zp)/-2.5)
		for i in range(len(sat)):
			xx = sat['x'].values[i]; yy = sat['y'].values[i]
			val = masking[int(yy),int(xx)]
			if np.isfinite(val):
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
				good = (ind[0] >= 0) & (ind[1] >= 0) & (ind[0] < self.data.shape[0]) & (ind[1] < self.data.shape[1])
				ind = ind[:,good]
				masking[ind[0],ind[1]] = np.nan
				dx = min(ind[1])
				dy = min(ind[0])
				inject[ind[0],ind[1]] += psf[ind[0]-dy,ind[1]-dx] * cflux*self.flux_factor + self.image_stats[1]
				#inject[ind[0],ind[1]] += psf[ind[0]-dy,ind[1]-dx] + self.image_stats[1]
		if self.mask is not None:
			#masking[self.mask>0] = self.image_stats[1] # set to the median
			self.newmask = (self.newmask + (inject > 0)) > 0
		replaced = np.nansum([masking,inject],axis=0)
		replaced[replaced == 0] = self.image_stats[1]
		self.replaced = replaced
		self.inject = inject


	def _save_image(self):
		newheader = deepcopy(self.header)
		newheader['SATCOR'] = (True,'Corrected sat stars')
		newheader['SATDATE'] = (date.today().isoformat(),'Date of correction')
		newheader['fluxfact'] = (self.flux_factor,'Correction to match catalog')
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

	def plot_flux_correction(self):
		plt.figure()
		plt.semilogy(self.calcat[self.band+'MeanPSFMag'],self._fitflux,'.',label='Image flux')
		plt.plot(self.calcat[self.band+'MeanPSFMag'],10**((self.calcat[self.band+'MeanPSFMag'].values-self.zp)/-2.5)*self.flux_factor,label='Corrected catalog flux')
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
		plt.imshow(self.data,vmax=vmax,vmin=vmin,origin='lower')
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
		ax = plt.gca()
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

def mask_rad_func(x, a=2.25512307e+03, b=3.30876377e-01, c=2.70001850e+01):
	rad = np.array(a * np.exp(-b * x) + c)
	rad = rad.astype(int)
	return rad


def _get_gaia(skycoord,rad):

	result = Vizier.query_region(skycoord, catalog=['I/355/gaiadr3'],
						 		 radius=Angle(rad, "deg"),column_filters={'Gmag':'<14'})
	if result is None:
		result = None
	else:
		result = result['I/355/gaiadr3'].to_pandas()
	return result

def _get_tyco(skycoord,rad):

	result = Vizier.query_region(skycoord, catalog=['J/PASP/120/1128/catalog'],
						 		 radius=Angle(rad, "deg"))
	if result is None:
		result = None
	else:
		result = result['J/PASP/120/1128/catalog'].to_pandas()
		result = result.rename(columns={'RAJ2000':'RA_ICRS','DEJ2000':'DE_ICRS'})	
	return result

def _get_bsc(skycoord,rad):
	result = Vizier.query_region(skycoord, catalog=['V/50/catalog'],
						 		 radius=Angle(rad, "deg"),column_filters={'Gmag':'<14'})
	if result is None:
		result = None
	else:
		try:
			result = result['V/50/catalog'].to_pandas()
			result = result.rename(columns={'RAJ2000':'RA_ICRS','DEJ2000':'DE_ICRS'})	
			c = SkyCoord(result.RA_ICRS.values,result.DE_ICRS.values, frame='icrs',unit=(u.hourangle, u.deg))
			result['RA_ICRS'] = c.ra.deg; result['DE_ICRS'] = c.dec.deg
		except:
			result = None
	return result
	
def find_unique2ps1(ps1_cat, viz_cat):
	radius_threshold = 2*u.arcsec
	
	coords_obs = SkyCoord(ra=ps1_cat.raMean, dec=ps1_cat.decMean, unit='deg')
	coords_viz = SkyCoord(ra=viz_cat['RA_ICRS'], dec=viz_cat['DE_ICRS'], unit='deg')
	idx, d2d, d3d = coords_viz.match_to_catalog_3d(coords_obs)
	sep_constraint = d2d >= radius_threshold
	viz_unique = viz_cat[sep_constraint]
 
	return viz_unique

def find_unique2viz(v1, v2):
	radius_threshold = 2*u.arcsec
	
	coords_v1 = SkyCoord(ra=v1['RA_ICRS'], dec=v1['DE_ICRS'], unit='deg')
	coords_v2 = SkyCoord(ra=v2['RA_ICRS'], dec=v2['DE_ICRS'], unit='deg')
	idx, d2d, d3d = coords_v2.match_to_catalog_3d(coords_v1)
	sep_constraint = d2d >= radius_threshold
	v2_unique = v2[sep_constraint]
 
	return v2_unique

def combine_close_sources(cat,distance=100):
	x = cat.x.values; y = cat.y.values
	dist = np.sqrt((x[:,np.newaxis] - x[np.newaxis,:])**2 + (y[:,np.newaxis] - y[np.newaxis,:])**2)
	far = dist > 100
	f = 10**((cat.mag.values-25)/-2.5)
	f = np.tile(f, (len(f), 1))
	f[far] = 0
	fn = np.nansum(f,axis=1)
	mn = -2.5*np.log10(fn) + 25
	cat['mag'] = mn
	return cat


import mastcasjobs

def search_refcat(ra, dec, search_size):
    query = """select n.distance as dstDegrees, r.*
               from fGetNearbyObjEq("""+str(ra)+','+str(dec)+","+str(search_size/2)+""") as n
               inner join refcat2 as r on n.objid=r.objid
               order by n.distance
    """

    jobs = mastcasjobs.MastCasJobs(context="HLSP_ATLAS_REFCAT2")
    results = jobs.quick(query, task_name="python atlas cone search")
    try:
        results = results.to_pandas()
        return(results)
    except:
        return None

def mag2flux(mag,zp):
    f = 10**(2/5*(zp-mag))
    return f

def PS1_to_TESS_mag(PS1):
    zp = 25
    g = mag2flux(PS1.gMeanPSFMag.values,zp)
    r = mag2flux(PS1.rMeanPSFMag.values,zp)
    i = mag2flux(PS1.iMeanPSFMag.values,zp)
    z = mag2flux(PS1.zMeanPSFMag.values,zp)
    y = mag2flux(PS1.yMeanPSFMag.values,zp)

    cr = 0.203; ci = 0.501; cz = 0.108
    cy = 0.186; p = -0.005

    t = (cr*r + ci*i + cz*z + cy*y)*(g/i)**p
    t = -2.5*np.log10(t) + zp
    PS1['tmag'] = t
    return PS1