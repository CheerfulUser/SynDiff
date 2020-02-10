# Import the TESS PRF modelling from DAVE
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path
sys.path.append('./dave/diffimg/')
import tessprf as prf

from astropy.visualization import (SqrtStretch, ImageNormalize)

from PS_image_download import *

from scipy import interpolate

from glob import glob
from astropy.io import fits
from astropy.wcs import WCS

from scipy.ndimage.filters import convolve

import tracemalloc
from scipy import signal

def Interp_PRF(X,Y,Camera,CCD):
	"""
	Create a TESS PSF interpolated to the PS scale from the PSF models.

	Inputs
	------
	X: float
		Centre column position
	Y: float
		Centre row position
	Camera: int 
		TESS camera
	CCD: int 
		TESS CCD 

	Returns
	-------
	kernal: array like
		The TESS PSF at the specified row and column position for a given 
		CCD for a Camera.
	"""
	pathToMatFile = './data/prf/'
	obj = prf.TessPrf(pathToMatFile)
	PRF = obj.getPrfAtColRow(X, Y, 1,Camera,CCD)
	x2 = np.arange(0,PRF.shape[1]-1,0.01075)
	y2 = np.arange(0,PRF.shape[0]-1,0.01075)

	x = np.arange(0,PRF.shape[1],1)
	y = np.arange(0,PRF.shape[0],1)
	X, Y = np.meshgrid(x,y)

	x=X.ravel()              #Flat input into 1d vector
	y=Y.ravel()

	z = PRF
	z = z.ravel()
	x = list(x[np.isfinite(z)])
	y = list(y[np.isfinite(z)])
	z = list(z[np.isfinite(z)])

	znew = interpolate.griddata((x, y), z, (x2[None,:], y2[:,None]), method='cubic')
	kernal = znew*(0.01075**2)
	return kernal

def Get_TESS_image(Path, Sector, Camera, CCD, Time = None):
    """
    Grabs a TESS FFI image from a directed path.
    Inputs
    ------
    Path: str
        Path to FFIs
    Sector: int
        Sector of the FFI
    Camera: int
        Camera of the FFI
    CCD: int
        CCD of the FFI
        
    Returns
    -------
    tess_image: array
        TESS image
    tess_wcs
        WCS of the TESS image
        
    Raises
    ------
    FileExistsError
        The file specified by the parameters does not exist.
        
    """
    if Time == None:
        File = "{Path}tess*-s{Sec:04d}-{Camera}-{CCD}*.fits".format(Path = Path, Sec = Sector, Camera = Camera, CCD = CCD)
    else:
        File = "{Path}tess{Time}-s{Sec:04d}-{Camera}-{CCD}*.fits".format(Path = Path, Time = Time, Sec = Sector, Camera = Camera, CCD = CCD)

    file = glob(File)
    if len(file) > 0:
        if (len(file) > 1):
            file = file[0]
        tess_hdu = fits.open(file)
        tess_wcs = WCS(tess_hdu[1].header)
        tess_image = tess_hdu[1].data
        return tess_image, tess_wcs
    else:
        raise FileExistsError("TESS file does not exist: '{}'".format(File))
        pass

def Print_snapshot():
	snapshot = tracemalloc.take_snapshot()
	top_stats = snapshot.statistics('lineno')

	print("[ Top 5 ]")
	for stat in top_stats[:5]:
		print(stat)
	return

def Plot_comparison(PSorig,PSconv,Downsamp = []):
	"""
	Makes plots for the convolution process.
	Inputs
	------
	PSorig: array like
		The original PS image
	PSconv: aray like
		The PS image convolved with TESS PSF
	Downsamp: array like
		PS image TESS PSF convolved and reduced to TESS resolution
	"""
	if len(Downsamp) == 0:
		plt.figure()
		
		plt.subplot(1, 2, 1)
		plt.title('PS original')
		plt.imshow(PSorig,origin='lower')#,vmax=1000)
		#plt.colorbar()

		plt.subplot(1, 2, 2)
		plt.title('PS convolved')
		plt.imshow(PSconv,origin='lower')#,vmax=1000)
		plt.tight_layout()
		#plt.colorbar()

		savename = 'Convolved_PS.pdf'
		plt.savefig(savename)
		return 'Plotted'
	else:
		plt.figure(figsize=(10, 4))
		
		norm = ImageNormalize(vmin=np.nanmin(PSorig)+0.1*np.nanmin(PSorig), vmax=np.nanmax(PSorig)-0.9*np.nanmax(PSorig), stretch=SqrtStretch())
		plt.subplot(1, 3, 1)
		plt.title('PS original')
		plt.imshow(PSorig,origin='lower',norm=norm)#,vmax=60000)
		#plt.colorbar()

		norm = ImageNormalize(vmin=np.nanmin(PSconv)+0.1*np.nanmin(PSconv), vmax=np.nanmax(PSconv)-0.1*np.nanmax(PSconv), stretch=SqrtStretch())
		plt.subplot(1, 3, 2)
		plt.title('PS convolved')
		plt.imshow(PSconv,origin='lower',norm=norm)#,vmax=1000)
		#plt.colorbar()

		norm = ImageNormalize(vmin=np.nanmin(Downsamp)+0.1*np.nanmin(Downsamp), vmax=np.nanmax(Downsamp)-0.1*np.nanmax(Downsamp), stretch=SqrtStretch())
		plt.subplot(1, 3, 3)
		plt.title('TESS resolution')
		plt.imshow(Downsamp,origin='lower',norm=norm)#,vmax=1000)
		plt.tight_layout()
		#plt.colorbar()

		savename = 'Convolved_PS_m82.pdf'
		plt.savefig(savename)
		return 'Plotted'

def Downsample(PSconv):
	"""
	Downsamples the PS image to the resolution of TESS.
	Inputs
    ------
    PSconv: array like
    	The PS image convolved with the TESS PSF
	Returns
	-------
	TESS_resolution: array like
		The PS image reduced to the TESS resolution
	"""
	PSpixel = 0.258 # arcseconds per pixel 
	TESSpixel = 21 # arcseconds per pixel 
	Scale = TESSpixel/PSpixel
	xnew = np.arange(PSconv.shape[1]/Scale)
	ynew = np.arange(PSconv.shape[0]/Scale)
	TESS_resolution = np.zeros((int(PSconv.shape[0]/Scale),int(PSconv.shape[1]/Scale)))
	for i in range(len(ynew)-1):
		ystart = int(i*Scale)
		yend = int(ystart + Scale)
		for j in range(len(xnew)-1):
			xstart = int(j*Scale)
			xend = int(xstart + Scale)
			TESS_resolution[i,j] = np.nansum(PSconv[ystart:yend,xstart:xend])
	return TESS_resolution
	
def PS_nonan(PS):
	PS_finite = np.copy(PS)
	PS_finite[np.isnan(PS)] = 3000000
	return PS_finite

def Run_convolution(Path,Sector,Camera,CCD,PSsize=1000,Downsamp=False,Plot=False):
	"""
	Wrapper function to convolve a PS image with the TESS PSF and return the convolved array.
	Inputs
    ------
    Path: str
        Path to FFIs
    Sector: int
        Sector of the FFI
    Camera: int
        Camera of the FFI
    CCD: int
        CCD of the FFI
    PSsize: int
    	Size in pixels of the PS image. 1 pixel corresponds to 0.024''.
        
    Saves
    -----
    test_PS_TESS: array
        PS image convolved with the appropriate TESS PSF
		
    Raises
    ------
    MemoryError
        There is not enough memory for this operation 

	"""
	tracemalloc.start()
	tess_image, tess_wcs = Get_TESS_image(Path,Sector,Camera,CCD)
	#Print_snapshot()

	x = tess_image.shape[1]/2
	y = tess_image.shape[0]/2
	kernal = Interp_PRF(x,y,Camera,CCD)
	#Print_snapshot()
	ra, dec = tess_wcs.all_pix2world(x,y,1)
	print('({},{})'.format(ra,dec))
	size = PSsize
	fitsurl = geturl(ra, dec, size=size, filters="i", format="fits")
	if len(fitsurl) > 0:
		fh = fits.open(fitsurl[0])
		ps = fh[0].data
		ps = PS_nonan(ps)
		try:
			#Print_snapshot()
			test = signal.fftconvolve(ps, kernal,mode='same')
			if Downsamp == True:
				down = Downsample(test)
			np.save('test_PS_TESS.npy',test)
			if Plot == True:
				if Downsamp == True:
					Plot_comparison(ps,test,Downsamp=down)
				else:
					Plot_comparison(ps,test)
		except MemoryError:
			raise MemoryError("The convolution is too large, try a smaller array.")

		return 'Convolved'
	else:
		return 'No PS images for RA = {}, DEC = {}'.format(ra,dec)

def Run_convolution_PS(Ra,Dec,Camera,CCD,PSsize=1000,Downsamp=False,Plot=False):
	"""
	Wrapper function to convolve a PS image with the TESS PSF and return the convolved array.
	Inputs
    ------
    Ra: float
        PS target RA
    Dec: float
        PS target DEC
    Camera: int
        Camera of the FFI
    CCD: int
        CCD of the FFI
    PSsize: int
    	Size in pixels of the PS image. 1 pixel corresponds to 0.024''.
        
    Saves
    -----
    test_PS_TESS: array
        PS image convolved with the appropriate TESS PSF
		
    Raises
    ------
    MemoryError
        There is not enough memory for this operation 

	"""

	x = 1000#tess_image.shape[1]/2
	y = 1000#tess_image.shape[0]/2
	kernal = Interp_PRF(x,y,Camera,CCD)
	#Print_snapshot()
	#ra, dec = tess_wcs.all_pix2world(x,y,1)
	print('({},{})'.format(Ra,Dec))
	size = PSsize
	fitsurl = geturl(Ra, Dec, size=size, filters="i", format="fits")
	if len(fitsurl) > 0:
		fh = fits.open(fitsurl[0])
		ps = fh[0].data
		ps = PS_nonan(ps)
		try:
			#Print_snapshot()
			test = signal.fftconvolve(ps, kernal,mode='same')
			if Downsamp == True:
				down = Downsample(test)
			np.save('test_PS_TESS.npy',test)
			if Plot == True:
				if Downsamp == True:
					Plot_comparison(ps,test,Downsamp=down)
				else:
					Plot_comparison(ps,test)
		except MemoryError:
			raise MemoryError("The convolution is too large, try a smaller array.")

		return 'Convolved'
	else:
		return 'No PS images for RA = {}, DEC = {}'.format(ra,dec)



