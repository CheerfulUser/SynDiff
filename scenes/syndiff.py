# Import the TESS PRF modelling from DAVE
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

import sys
sys.path
sys.path.append(PACKAGEDIR + '/dave/diffimg/')
#print(sys.path)
import tessprf as prf



from PS_image_download import *

from scipy import interpolate

from glob import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.visualization import (SqrtStretch, ImageNormalize)

from scipy.ndimage.filters import convolve
from scipy.interpolate import RectBivariateSpline
from scipy import signal

import tracemalloc

from scipy.ndimage import  rotate


from mpl_toolkits.axes_grid1 import make_axes_locatable

def Get_PRF(Row,Col,Camera,CCD):
	"""
	Get the TESS PRF model from dave.

	-------
	Inputs-
	-------
		Row 	float	row position of the object
		Col 	float 	column position of the object
		Camera 	int 	TESS camera  range 1-4
		CCD 	ind 	TESS camera CCD range 1-4
	-------
	Output-
	-------
		PRF 	array 	PRF model for the given position and detector
	"""
	pathToMatFile = PACKAGEDIR + '/data/prf/'
	obj = prf.TessPrf(pathToMatFile)
	PRF = obj.getPrfAtColRow(Col, Row, 1,Camera,CCD)
	#PRF = PRF / np.nansum(PRF)
	return PRF

def Interp_PRF(Row,Col,Camera,CCD, Scale, Method = 'RBS'):
	"""
	Create a TESS PSF interpolated to the PS scale from the PSF models.
	-------
	Inputs-
	-------
	Row 		float 	Centre row position
	Col 		float 	Centre column position
	Camera 		int 	TESS camera
	CCD 		int 	TESS CCD 
	
	-------
	Option-
	-------
	Method 		str 	Interpolation method, only 'RBS' and 'grid'

	-------
	Output-
	-------
	kernal 		array 	The TESS PSF at the specified row and column 
						position for a given CCD for a Camera.
	"""
	pathToMatFile = PACKAGEDIR + '/data/prf/'
	obj = prf.TessPrf(pathToMatFile)
	PRF = obj.getPrfAtColRow(Col, Row, 1,Camera,CCD)
	PRF = rotate(PRF,-90)
	norm = np.nansum(PRF)
	x2 = np.arange(0,PRF.shape[0]-1, 1/Scale)
	y2 = np.arange(0,PRF.shape[1]-1, 1/Scale)

	x = np.arange(0,PRF.shape[0],1)
	y = np.arange(0,PRF.shape[1],1)

	if Method == 'griddata':
		X, Y = np.meshgrid(x,y)
		x=X.ravel()			  #Flat input into 1d vector
		y=Y.ravel()

		z = PRF
		z = z.ravel()
		x = list(x[np.isfinite(z)])
		y = list(y[np.isfinite(z)])
		z = list(z[np.isfinite(z)])

		znew = interpolate.griddata((x, y), z, (x2[None,:], y2[:,None]), method='cubic')
		kernal = znew

	if Method == 'RBS':
		func = RectBivariateSpline(x,y,PRF)
		kernal = func(x2,y2)
	# normalise kernal to 1 since zerpoint accounts for losses 
	kernal = kernal / np.nansum(kernal)

	return kernal

def Get_TESS(RA,DEC,Size,Sector=None):
	'''
	Use the lightkurve mast query to get FFI cutouts of a given position
	'''
	c = SkyCoord(ra=float(RA)*u.degree, dec=float(DEC) *
				 u.degree, frame='icrs')
	
	tess = lk.search_tesscut(c,sector=Sector)
	tpf = tess.download(cutout_size=Size)
	
	return tpf

def Get_TESS_local(Path, Sector, Camera, CCD, Time = None):
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

def Get_Gaia(tpf, magnitude_limit = 18):
	"""
	Get the coordinates and gia mag of all Gaia sources in the field of view.

	-------
	Inputs-
	-------
		tpf 				class 	target pixel file lightkurve class
		magnitude_limit 	float 	cutoff for Gaia sources
	
	--------
	Outputs-
	--------
		coords 	array	coordinates of sources
		Gmag 	array 	Gmags of sources
	"""
	c1 = SkyCoord(tpf.ra, tpf.dec, frame='icrs', unit='deg')
	# Use pixel scale for query size
	pix_scale = 4.0  # arcseconds / pixel for Kepler, default
	if tpf.mission == 'TESS':
		pix_scale = 21.0
	# We are querying with a diameter as the radius, overfilling by 2x.
	from astroquery.vizier import Vizier
	Vizier.ROW_LIMIT = -1
	result = Vizier.query_region(c1, catalog=["I/345/gaia2"],
								 radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))
	no_targets_found_message = ValueError('Either no sources were found in the query region '
										  'or Vizier is unavailable')
	too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
	if result is None:
		raise no_targets_found_message
	elif len(result) == 0:
		raise too_few_found_message
	result = result["I/345/gaia2"].to_pandas()
	
	result = result[result.Gmag < magnitude_limit]
	if len(result) == 0:
		raise no_targets_found_message
	radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
	coords = tpf.wcs.all_world2pix(radecs, 0) ## TODO, is origin supposed to be zero or one?
	Gmag = result['Gmag'].values
	#Jmag = result['Jmag']
	ind = (((coords[:,0] >= -0.5*tpf.shape[1]) & (coords[:,1] >= -0.5*tpf.shape[2])) & 
		   ((coords[:,0] < 1.5*tpf.shape[1]) & (coords[:,1] < 1.5*tpf.shape[2])))
	coords = coords[ind]
	Gmag = Gmag[ind]
	#Jmag = Jmag[ind]
	return coords, Gmag

def Gaia_scene(Ra,Dec,Size,Maglim= 19,Bkg_limit = 20.5, Zeropoint = 20.44, 
				Scale = 100,Interpolate = False, FFT = False,Plot= False, 
				Save = None):
	"""
	Create a simulated TESS image using Gaia sources. 

	-------
	Inputs-
	-------
		Ra 				float 	RA of image centre 
		DEC 			float 	Dec of image centre 
		Size 			int 	Size of the TESS image in pixels
		Maglim 			float 	Magnitude limit for Gaia sources
		Bkg_lim 		float 	TESS limiting magnitude 
		Zeropoint 		float 	TESS magnitude zeropoint 
		Scale 			int 	Interpolation scale size 
	
	--------
	Options-
	--------
		Interpolate 	bool	Interpolate the TESS PRF to a scale specified by parameter 'Scale'
		Plot 			bool 	Plot the complete scene and TESS image 
		Save 			str 	Save path for figure

	-------
	Output-
	-------
		soures 			array 	Array of simulated TESS images for each Gaia sourcetes

	"""
	tpf = Get_TESS(Ra,Dec,Size)
	# pos returned as column row 
	pos, gmag = Get_Gaia(tpf,magnitude_limit=Maglim)
	col = pos[:,0]
	row = pos[:,1]
	
	Tmag = gmag - 0.5
	tcounts = 10**(-2/5*(Tmag - Zeropoint))
	bkg = 10**(-2/5*(Bkg_limit - Zeropoint))

	sources = np.zeros((len(pos),tpf.shape[1],tpf.shape[2])) + bkg
	for i in range(len(pos)):
		if Interpolate:
			template = np.zeros((2*tpf.shape[1]*Scale,2*tpf.shape[2]*Scale))
			#print('template shape ',template.shape)
			offset1 = int(0.5 * tpf.shape[1] * Scale)
			offset2 = int(0.5 * tpf.shape[2] * Scale)
			#print(np.nansum(template))
			kernal = Interp_PRF(row[i] + tpf.row, col[i] + tpf.column,tpf.camera,tpf.ccd,Scale)
			#print(np.nansum(kernal))
			if FFT:
				template[int(row[i]*Scale + offset1),int(col[i]*Scale+ offset2)] = tcounts[i]
				template = signal.fftconvolve(template, kernal, mode='same')
			else:
				optics = kernal * tcounts[i]
				r = int(row[i]*Scale + offset1)
				c = int(col[i]*Scale + offset2)
				template = Add_convolved_sources(r,c,optics,template)
			#print(np.nansum(template))
			template = template[offset1:int(3*offset1),offset2:int(3*offset2)]
			#print('template shape ',template.shape)
			sources[i] = Downsample(template,Scale)

		else:
			template = np.zeros((2*tpf.shape[1],2*tpf.shape[2]))
			kernal = Get_PRF(row[i] + tpf.row, col[i] + tpf.column,tpf.camera,tpf.ccd)
			kernal = kernal / np.nansum(kernal)
			#print(template.shape)
			offset1 = int(0.5 * tpf.shape[1])
			offset2 = int(0.5 * tpf.shape[2])
			template[int(row[i] + offset1),int(col[i] + offset2)] = tcounts[i]
			template = signal.fftconvolve(template, kernal, mode='same')
			template = template[offset1:int(3*offset1),offset2-1:int(3*offset2-1)]
		
			sources[i] = template
	if Plot:
		gaia = np.nansum(sources,axis=0)
		#gaia = rotate(np.flipud(gaia*10),-90)
		plt.figure(figsize=(8,4))
		plt.subplot(1,2,1)
		plt.title('Gaia scene')
		norm = ImageNormalize(vmin=np.nanmin(gaia), 
							  vmax=np.nanmax(gaia), stretch=SqrtStretch())
		im = plt.imshow(gaia,origin='lower', norm = norm)
		plt.xlim(-0.5, Size-0.5)
		plt.ylim(-0.5, Size-0.5)
		plt.plot(pos[:,0],pos[:,1],'r.')
		ax = plt.gca()
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(im, cax=cax)


		plt.subplot(1,2,2)	
		plt.title('TESS image')
		tess = np.nanmedian(tpf.flux,axis=0)
		norm = ImageNormalize(vmin=np.nanmin(tess), 
							  vmax=np.nanmax(tess), stretch=SqrtStretch())
		im = plt.imshow(tess,origin='lower',norm = norm)
		plt.xlim(-0.5, Size-0.5)
		plt.ylim(-0.5, Size-0.5)
		plt.plot(pos[:,0],pos[:,1],'r.')
		ax = plt.gca()
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(im, cax=cax)

		plt.tight_layout()
		#if type(Save) != type(None):
		#	plt.savefig(Save)

	return sources

def Add_convolved_sources(Row, Col, Optics,Template):
	"""
	An ugly function that inserts a small array into a larger array.
	With this fft is not needed for single objects.
	
	-------
	Inputs-
	-------
	Row 		int  	Row of source
	Col 		int  	Column of source
	Optics 		array 	Small array to inject
	Template 	array 	Large array to be get injected 

	-------
	Output-
	-------
	Template 	array 	Large array with small array injected 

	"""
	start1 = int(Row - Optics.shape[0]/2)
	end1 = int(Row + Optics.shape[0]/2)
	
	if start1 < 0:
		o_start1 = abs(start1)
		start1 = 0
	else:
		o_start1 = 0

	if end1 > Template.shape[0]:
		o_end1 = Optics.shape[0]-abs(end1 - (Template.shape[0]))
		end1 = Template.shape[0]
	else:
		o_end1 = Optics.shape[0]

	start2 = int(Col - Optics.shape[1]/2)
	end2 = int(Col + Optics.shape[1]/2)

	if start2 < 0:
		o_start2 = abs(start2)
		start2 = 0
	else:
		o_start2 = 0

	if end2 > Template.shape[1]:
		o_end2 = Optics.shape[1]-abs(end2 - (Template.shape[1]))
		end2 = Template.shape[1]
	else:
		o_end2 = Optics.shape[1]
		
	#print(o_start1,o_end1)
	#print(o_start2,o_end2)
	#print(template[start1:end1,start2:end2].shape)
	#print('optics', optics[o_start1:o_end1,o_start2:o_end2].shape)
	Template[start1:end1,start2:end2] = Optics[o_start1:o_end1,o_start2:o_end2]
	
	return Template


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

def Downsample(Image,Scale):
	"""
	Downsamples an image to the resolution specified by 'Scale'.
	-------
	Inputs-
	-------
	Image 	 	array 		High resolution image
	Scale 		int 		Scale factor 

	--------
	Returns-
	--------
	down 		array 		Downsampled image
	"""
	#PSpixel = 0.258 # arcseconds per pixel 
	#TESSpixel = 21 # arcseconds per pixel 
	#Scale = TESSpixel/PSpixel
	xnew = np.arange(Image.shape[1]/Scale)
	ynew = np.arange(Image.shape[0]/Scale)
	down = np.zeros((int(Image.shape[0]/Scale),int(Image.shape[1]/Scale)))
	#print('down ', down.shape)
	for i in range(len(ynew)):
		ystart = int(i*Scale)
		yend = int(ystart + Scale)
		for j in range(len(xnew)):
			xstart = int(j*Scale)
			xend = int(xstart + Scale)
			down[i,j] = np.nansum(Image[ystart:yend,xstart:xend])
	#print('down ', down.shape)
	return down
	
def PS_nonan(PS):
	'''
	Removes nans from PS images. Very basic and must be improved.
	------
	Input-
	------
		PS 			array 	Image array
	
	-------
	Output-
	-------
		PS_finite	array 	Image
	'''
	PS_finite = np.copy(PS)
	PS_finite[np.isnan(PS)] = 0
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



