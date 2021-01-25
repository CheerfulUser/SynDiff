import numpy as np
from scipy.signal import savgol_filter
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
import sigmacut
from scipy.ndimage.filters import convolve
from scipy import interpolate
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.time import Time
from scipy.ndimage import shift
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import os

def Save_space(Save):
    """
    Creates a path if it doesn't already exist.
    """
    try:
        if not os.path.exists(Save):
            os.makedirs(Save)
    except FileExistsError:
        pass


def sigma_mask(data,error= None,sigma=3,Verbose= False):
	if type(error) == type(None):
		error = np.zeros(len(data))
	
	calcaverage = sigmacut.calcaverageclass()
	calcaverage.calcaverage_sigmacutloop(data,Nsigma=sigma
										 ,median_firstiteration=True,saveused=True)
	if Verbose:
		print("mean:%f (uncertainty:%f)" % (calcaverage.mean,calcaverage.mean_err))
	return calcaverage.clipped

'''
change to have average time in the reference image
'''

def Reference_image(files,FITS=False):
	summed = np.zeros(len(files)) * np.nan
	for i in range(len(files)):
		hdu = fits.open(files[i])
		data = hdu[1].data
		wcs = WCS(hdu[1].header)
		cut = Cutout2D(data,(1024+44,1024),2048,wcs=wcs)
		data = cut.data 
		wcs = cut.wcs
		data[data <= 0] = np.nan
		if np.nansum(abs(data)) > 0:
			summed[i] = np.nansum(abs(data))
	
	lim = np.percentile(summed[np.isfinite(summed)],5)
	ind = np.where((summed < lim))[0]
	good = files[ind]
	goods = np.zeros((len(good),2048,2048))
	var = np.zeros((len(good),2048,2048))
	mjd = np.zeros(len(good))
	i = 0
	for g in good:
		hdu = fits.open(g)
		data = hdu[1].data
		wcs = WCS(hdu[1].header)
		cut = Cutout2D(data,(1024+44,1024),2048,wcs=wcs)
		data = cut.data 
		wcs = cut.wcs
		goods[i] = data 

		e = hdu[2].data
		cut = Cutout2D(e,(1024+44,1024),2048)
		data = cut.data 
		var[i] = data**2  

		jd = hdu[1].header['TSTART'] + hdu[1].header['BJDREFI']
		mjd[i] = Time(jd, format='jd', scale='tdb').mjd

		i += 1
	ref = np.nanmedian(goods,axis=0)
	var = np.nanmedian(var,axis=0)
	hdu[1].header['MJD'] = (np.nanmean(mjd), 'stacked')

	if FITS:
		ref_fits = deepcopy(hdu)
		ref_fits[1].data = ref
		ref_fits[1].header.update(wcs.to_header())
		ref_fits[2].data = err
		ref_fits[2].header.update(wcs.to_header())
		return ref_fits
	else:
		return ref, np.sqrt(var), wcs, hdu

def Source_mask(data, grid=True):
	
	if grid:
		data[data<0] = np.nan
		data[data >= np.percentile(data,95)] =np.nan
		grid = np.zeros_like(data)
		size = 32
		for i in range(grid.shape[0]//size):
			for j in range(grid.shape[1]//size):
				section = data[i*size:(i+1)*size,j*size:(j+1)*size]
				section = section[np.isfinite(section)]
				lim = np.percentile(section,1)
				grid[i*size:(i+1)*size,j*size:(j+1)*size] = lim
		thing = data - grid
	else:
		thing = data
	ind = np.isfinite(thing)
	mask = ((thing <= np.percentile(thing[ind],80,axis=0)) |
		   (thing <= np.percentile(thing[ind],10))) * 1.0
		
	return mask 

def Smooth_bkg(data, extrapolate = True, quality = False):
	
	data[data == 0] = np.nan
	x = np.arange(0, data.shape[1])
	y = np.arange(0, data.shape[0])
	arr = np.ma.masked_invalid(data)
	xx, yy = np.meshgrid(x, y)
	#get only the valid values
	x1 = xx[~arr.mask]
	y1 = yy[~arr.mask]
	newarr = arr[~arr.mask]

	estimate = interpolate.griddata((x1, y1), newarr.ravel(),
							  (xx, yy),method='linear')
	bitmask = np.zeros_like(data,dtype=int)
	bitmask[np.isnan(estimate)] = 128 | 4
	nearest = interpolate.griddata((x1, y1), newarr.ravel(),
							  (xx, yy),method='nearest')
	if extrapolate:
		estimate[np.isnan(estimate)] = nearest[np.isnan(estimate)]
	
	estimate = gaussian_filter(estimate,9)

	return estimate, bitmask
	
def Strap_bkg(data):
	
	ind = np.where(np.nansum(abs(data),axis=0)>0)[0]
	strap_bkg = np.zeros_like(data)
	for col in ind:
		x = np.arange(0,data.shape[1])
		y = data[:,col].copy()
		finite = np.isfinite(y)
		if len(y[finite]) > 5:
			finite = np.isfinite(y)
			bad = sigma_mask(y[finite],sigma=2)
			finite = np.where(finite)[0]
			y[finite[bad]] = np.nan
			finite = np.isfinite(y)
			#regressionLine = np.polyfit(x[finite], y[finite], 3)
			fit = UnivariateSpline(x[finite], y[finite])
			fit.set_smoothing_factor(1500)
			#p = interp1d(x[finite], y[finite],bounds_error=False,fill_value=np.nan,kind='cubic')
			#p = np.poly1d(regressionLine)
			p = fit(x)
			finite = np.isfinite(p)
			smooth =savgol_filter(p[finite],13,3)
			p[finite] = smooth

			thingo = y - p
			finite = np.isfinite(thingo)
			bad = sigma_mask(thingo[finite],sigma=2)
			finite = np.where(finite)[0]
			y[finite[bad]] = np.nan
			finite = np.isfinite(y)
			#regressionLine = np.polyfit(x[finite], y[finite], 3)
			#p = np.poly1d(regressionLine)
			#p = interp1d(x[finite], y[finite],bounds_error=False,fill_value=np.nan,kind='cubic')
			fit = UnivariateSpline(x[finite], y[finite])
			fit.set_smoothing_factor(1500)
			p = fit(x)
			finite = np.isfinite(p)
			smooth =savgol_filter(p[finite],13,3)
			p[finite] = smooth
			strap_bkg[:,col] = p
	
	return strap_bkg

def Background(Data,Mask):
	mask = deepcopy(Mask)
	data = deepcopy(Data)

	strap_mask = np.zeros_like(data)
	straps = pd.read_csv('tess_straps.csv')['Column'].values
	strap_mask[:,straps-1] = 1
	big_strap = convolve(strap_mask,np.ones((3,3))) > 0
	big_mask = convolve((mask==0)*1,np.ones((8,8))) > 0

	masked = data * ((big_mask==0)*1) * ((big_strap==0)*1)
	masked[masked == 0] = np.nan
	bkg_smooth, bitmask = Smooth_bkg(masked, extrapolate = True, quality = True)
	round1 = data - bkg_smooth
	round2 = round1 * ((big_strap==1)*1) * ((big_mask==1)*1)
	round2[round2 == 0] = np.nan
	strap_bkg = Strap_bkg(round2)

	return strap_bkg + bkg_smooth, bitmask

def Make_fits(hdu,data,wcs,name, noise = False, stacked = False):
	newhdu = fits.PrimaryHDU(data, header = hdu[1].header)
	gain = np.nanmean([newhdu.header['GAINA'],newhdu.header['GAINB'],newhdu.header['GAINC'],newhdu.header['GAIND']])

	newhdu.header['NAXIS1'] = 2048
	newhdu.header['NAXIS2'] = 2048
	newhdu.header['BACKAPP'] = 'T'
	newhdu.header['NOISEIM'] = 1
	newhdu.header['MASKIM'] = 1
	newhdu.header['GAIN'] = (gain, '[electrons/count] Average CCD output gain')
	newhdu.header['PIXSCALE'] = 21 # pixel scale in arcsec / pix
	newhdu.header['SW_PLTSC'] = 21 # pixel scale in arcsec / pix
	newhdu.header['PHOTCODE'] = 0x9500
	newhdu.header['SATURATE'] = 65535
	newhdu.header['STACK'] = stacked

	#newhdu.header.update(wcs.to_header())
	# shift the reference coord for the pipeline wcs by 44 columns 
	newhdu.header['CRPIX1'] = newhdu.header['CRPIX1'] - 44 
	if noise:
		bscale = 0.1
		bzero = 3276.80
	else:
		bscale = 1.0
		bzero = 3276.80
	# set max and min of bscale
	maxval = 32767.0 * bscale + bzero
	minval = -32768.0 * bscale + bzero

	# overflow checking 
	toohigh = data > maxval
	data[toohigh] = maxval
	toolow = data<minval
	data[toolow] = minval


	newhdu.scale('int16', bscale=bscale,bzero=bzero)
	newhdu.header['BSCALE'] = bscale
	newhdu.header['BZERO'] = bzero
	newhdu.writeto(name,overwrite=True)
	return 

def figures(data, bkg, err, save):

	plt.figure(figsize=(8,8))
	plt.subplot(2,2,1)
	plt.title('Raw')
	im = plt.imshow(data,origin='',vmin=np.percentile(data,10),
				vmax=np.percentile(data,90))
	ax = plt.gca()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)

	plt.subplot(2,2,2)
	plt.title('Error')
	im = plt.imshow(err,origin='',vmin=np.percentile(err,10),
				vmax=np.percentile(err,90))
	ax = plt.gca()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)

	plt.subplot(2,2,3)
	plt.title('Background')
	im = plt.imshow(bkg,origin='',vmin=np.percentile(bkg,10),
				vmax=np.percentile(bkg,90))
	ax = plt.gca()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)


	plt.subplot(2,2,4)
	sub = data - bkg
	plt.title('Subbed')
	im = plt.imshow(sub,origin='',vmin=np.percentile(sub,10),
				vmax=np.percentile(sub,90))
	ax = plt.gca()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	plt.tight_layout()
	plt.savefig(save)
	plt.close()



def FFI_bkg(datapath,save):
	files = np.array(glob(datapath + '*.fits'))
	sector = files[0].split('-')[1]
	cam = files[0].split('-')[2]
	ccd = files[0].split('-')[3]
	ref, err, wcs, hdu = Reference_image(files)
	
	mask = Source_mask(ref)

	bkg, bitmask = Background(ref,mask)
	saturation = ref > (4.8E4 - 500)

	bitmask[saturation] = bitmask[saturation] | (128 | 2)
	ref = ref - bkg
	ref += 500 # add a pedastal value 
	bitmask[ref < 0] = bitmask[ref < 0] | (128 | 4)
	skysig = np.nanmedian(np.nanstd(ref*convolve(mask,np.ones((3,3)))))
	skyadu = np.nanmedian(np.nanmedian(ref*convolve(mask,np.ones((3,3)))))
	hdu[1].header['SKYADU'] = (skyadu, 'median sky')
	hdu[1].header['SKYSIG'] = (skyadu, 'median sky noise')
	directory = save + 'tmpl/' + str(int(cam) * int(ccd)) + '/' 
	Save_space(directory)
	name = directory + sector + '_stack_' + str(int(cam) * int(ccd)) + '.pdf'
	figures(ref,bkg,err,name)
	name = directory + sector + '_stack_' + str(int(cam) * int(ccd)) + '.fits.fz'
	Make_fits(hdu,ref,wcs,name,stacked=True)
	name = directory + sector + '_stack_' + str(int(cam) * int(ccd)) + '.bkg.fits.fz'
	Make_fits(hdu,bkg,wcs,name,stacked=True)
	name = directory + sector + '_stack_' + str(int(cam) * int(ccd)) + '.mask.fits.fz'
	Make_fits(hdu,bitmask,wcs,name,stacked=True)
	name = directory + sector + '_stack_' + str(int(cam) * int(ccd)) + '.noise.fits.fz'
	Make_fits(hdu,err,wcs,name,True,stacked=True)

	#Make_fits(hdu,mask,wcs,name)
	for file in files:
		date = file.split('tess')[-1].split('-')[0]
		hdu = fits.open(file)
		gain = np.nanmean([hdu[1].header['GAINA'],hdu[1].header['GAINB'],
							hdu[1].header['GAINC'],hdu[1].header['GAIND']])
		data = hdu[1].data
		err = hdu[2].data 
		wcs = WCS(hdu[1].header)
		cut = Cutout2D(data,(1024+44,1024),2048,wcs=wcs)
		data = cut.data
		wcs = cut.wcs
		err = Cutout2D(err,(1024+44,1024),2048).data
		if np.nansum(data) > 1000:
			bkg, bitmask = Background(data,mask)
			saturation = data > (4.8E4 - 500)
			bitmask[saturation] = bitmask[saturation] | (128 | 2)
			sub = data - bkg
			skysig = np.nanmedian(np.nanstd(sub*convolve(mask,np.ones((3,3)))))
			skyadu = np.nanmedian(np.nanmedian(sub*convolve(mask,np.ones((3,3)))))
			hdu[1].header['SKYADU'] = (skyadu, 'median sky')
			hdu[1].header['SKYSIG'] = (skysig, 'median sky noise')

			jd = hdu[1].header['TSTART'] + hdu[1].header['BJDREFI']
			hdu[1].header['MJD'] = Time(jd, format='jd', scale='tdb').mjd
			sub += 500 # add a pedastal value 
			bad_sub = sub < 0
			bitmask[bad_sub] = bitmask[bad_sub] | (128 | 4)
			directory = save + sector + '/' + str(int(cam) * int(ccd)) + '/'
			Save_space(directory)
			name = directory + sector + '_' + date + '_' + str(int(cam) * int(ccd)) + '.pdf'
			figures(data,bkg,err,name)
			
			name = directory + sector + '_' + date + '_' + str(int(cam) * int(ccd)) + '.fits.fz'
			Make_fits(hdu,sub,wcs,name)
			name = directory + sector + '_' + date + '_' + str(int(cam) * int(ccd)) + '.bkg.fits.fz'
			Make_fits(hdu,bkg,wcs,name)
			err[np.isnan(err)] = 0
			name = directory + sector + '_' + date + '_' + str(int(cam) * int(ccd)) + '.noise.fits.fz'
			Make_fits(hdu,err,wcs,name,True)
			name = directory + sector + '_' + date + '_' + str(int(cam) * int(ccd)) + '.mask.fits.fz'
			Make_fits(hdu,bitmask,wcs,name)
			hdu.close()

	return 'Done'