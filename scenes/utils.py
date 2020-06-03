import numpy as np
from photutils import centroid_com
from scipy.ndimage import shift
from astropy.wcs import WCS
import matplotlib.pyplot as plt

def pix2coord(x, y, mywcs):
	"""
	Calculates RA and DEC from the pixel coordinates
	"""
	wx, wy = mywcs.all_pix2world(x, y, 1)
	return np.array([float(wx), float(wy)])

def coord2pix(x, y, mywcs):
	"""
	Calculates RA and DEC from the pixel coordinates
	"""
	wx, wy = mywcs.all_world2pix(x, y, 1)
	return np.array([float(wx), float(wy)])

def Gaussian2D(Size, FWHM = 130, Center=None):
	""" 
	Make a 2D Gaussian to act as the intra-pixel response function. 
	Currenty the FWHM is set to mimic the figs in https://arxiv.org/pdf/1806.07430.pdf.
	A better value is needed.
	
	-------
	Inputs-
	-------
		Size  int size of the array 
		FWHM  float full width half max of the array 
		Center  array/list position of the gaussian's center 
	-------
	Output-
	-------
		gauss array 2D Gaussian
	"""

	x = np.arange(0, Size, 1, float)
	y = x[:,np.newaxis]

	if Center is None:
		x0 = y0 = Size // 2
	else:
		x0 = Center[0]
		y0 = Center[1]
	gauss = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / FWHM**2)
	return gauss


def Isolated_stars(pos,Tmag,flux,Median,sources, Distance = 7, Mag = 16):
	"""
	Find isolated stars in the scene.
	"""
	#pos, Tmag = sd.Get_PS1(tpf,magnitude_limit=18)
	pos_shift = pos -.5
	ind = ((Distance//2< pos_shift[:,0]) & (pos_shift[:,0]< flux.shape[1]-Distance//2) & 
		  (Distance//2< pos_shift[:,1]) & (pos_shift[:,1]< flux.shape[1]-Distance//2) &
		  (Tmag < Mag))
	if ~ind.any():
		raise ValueError('No sources brighter than {} Tmag.'.format(Mag))
	p = pos_shift[ind,:]
	distance= np.zeros([len(p),len(p)])
	for i in range(len(p)):
		distance[i] = np.sqrt((p[i,0] - p[:,0])**2 + (p[i,1] - p[:,1])**2)
	distance[distance==0] = np.nan
	mins = np.nanmin(distance,axis=1)
	iso = p[mins > Distance]
	iso = iso.astype('int')
	iso_s = sources[ind]
	iso_s = iso_s[mins > Distance]
	median = Median
	median[median<0] = 0
	if len(iso)> 0:
		clips = []
		time_series = []
		if (Distance % 2) ==0:
			d = Distance - 1
		else:
			d = Distance
		u = d//2 +1
		l = d //2 
		
		for i in range(len(iso)):
			clips += [[iso_s[i,iso[i,1]-l:iso[i,1]+u,iso[i,0]-l:iso[i,0]+u],
					 median[iso[i,1]-l:iso[i,1]+u,iso[i,0]-l:iso[i,0]+u]]]
			time_series += [flux[:,iso[i,1]-l:iso[i,1]+u,iso[i,0]-l:iso[i,0]+u]]
		#print(clips)
		clips=np.array(clips)
		time_series=np.array(time_series)
	else:
		raise ValueError('No stars brighter than {} Tmag and isolated by {} pix. Concider lowering brightness.'.format(Mag,Distance))
	return iso, clips, time_series

def Centroids(Stars,References,Trim=0,Plot=False):
	"""
	Calculate the centroid offsets for a set of reference stars.
	"""
	flags = np.zeros((Stars.shape[0]))
	centroids = np.zeros((Stars.shape[0],Stars.shape[1],2))
	for i in range(len(Stars)):
		star = Stars[i]
		lc = np.nansum(star,axis=(1,2))
		lc[lc==0] = np.nan
		lc = lc/np.nanmedian(lc)
		bads = np.nansum(lc>1.2)
		if bads >= 10:
			flags[i] = 1
		for j in range(len(star)):
			if Trim == 0:
				c = centroid_com(star[j])
				ref = centroid_com(References[i])
			else:
				c = centroid_com(star[j,Trim:-Trim,Trim:-Trim])
				ref = centroid_com(References[i,Trim:-Trim,Trim:-Trim])
			c = c - ref
			centroids[i,j] = c
		if Plot:
			Plot_centroids(centroids[i],star,i)
	return centroids, flags

def Plot_centroids(Centoids,Star,Num,Save=False):
	x = np.arange(0,len(Centoids))
	plt.figure()
	plt.subplot(221)
	plt.scatter(x,Centoids[:,0],marker='.',c=x,alpha = 1)
	plt.ylabel('$\Delta$Column')
	plt.xlabel('Frame')
	plt.subplot(222)
	plt.scatter(x,Centoids[:,1],marker='.',c=x,alpha = 1)
	plt.ylabel('$\Delta$Row')
	plt.xlabel('Frame')
	plt.subplot(223)
	im = plt.scatter(Centoids[:,0],Centoids[:,1],marker='.',c=x,alpha = 1)
	plt.ylabel('$\Delta$Row')
	plt.xlabel('$\Delta$Column')
	plt.subplot(224)
	lc = np.nansum(Star,axis=(1,2))
	lc[lc==0] = np.nan
	lc = lc/np.nanmedian(lc)
	bads = np.nansum(lc>1.2)
	if bads >= 10:
		plt.plot(lc/np.nanmedian(lc),'r-')
	else:
		plt.plot(lc/np.nanmedian(lc),'g-')
	#plt.axhline(np.nanmedian(lc),ls='--',c='k')
	plt.ylabel('Counts')
	plt.xlabel('Frame (Bad frames = {})'.format(bads) )
	plt.suptitle('Reference star {}'.format(Num))
	plt.subplots_adjust(wspace=.35,hspace=.4)
	
	if Save:
		plt.savefig('./Centroids_Star_{}.pdf'.format(Num))
	#plt.tight_layout()





def Shift_images(Offset,Data,median=False):
	"""
	Shifts data by the values given in offset. Breaks horribly if data is all 0.

	"""
	shifted = Data.copy()
	data = Data.copy()
	data[data<0] = 0
	for i in range(len(data)):
		if np.nansum(data[i]) > 0:
			shifted[i] = sd.shift(data[i],[-Offset[i,1],-Offset[i,0]],mode='nearest',order=3)
	return shifted

def Shift_median(Offset,Med):
	"""
	Shifts data by the values given in offset. Breaks horribly if data is all 0.

	"""
	shifted = np.zeros((len(Offset),Med.shape[0],Med.shape[1]))
	Med[np.isnan(Med)] = 0
	Med[Med<0] = 0
	for i in range(len(Offset)):
		shifted[i] = sd.shift(Med,[Offset[i,1],Offset[i,0]],mode='nearest',order=3)

	return shifted