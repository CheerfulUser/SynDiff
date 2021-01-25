import numpy as np
from photutils import centroid_com
from scipy.ndimage import shift
from astropy.wcs import WCS
import matplotlib.pyplot as plt

import scipy
from scipy.signal import savgol_filter

from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip


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



def Smooth_motion(Centroids,tpf):
    split = np.where(np.diff(tpf.astropy_time.mjd) > 0.5)[0][0] + 1
    smoothed = np.zeros_like(Centroids) * np.nan
    # ugly, but who cares
    ind1 = np.nansum(tpf.flux[:split],axis=(1,2))
    ind1 = np.where(ind1 != 0)[0]
    ind2 = np.nansum(tpf.flux[split:],axis=(1,2))
    ind2 = np.where(ind2 != 0)[0] + split
    smoothed[ind1,0] = savgol_filter(Centroids[ind1,0],51,3)
    smoothed[ind2,0] = savgol_filter(Centroids[ind2,0],51,3)

    smoothed[ind1,1] = savgol_filter(Centroids[ind1,1],51,3)
    smoothed[ind2,1] = savgol_filter(Centroids[ind2,1],51,3)
    return smoothed

def Plot_centroids(TPF,Shifts, Smoothed, Save = None):
    t = TPF.astropy_time.mjd
    meds = Shifts.copy()
    smooth = Smoothed.copy()
    plt.figure()
    plt.plot(t,meds[:,0],'.',label='Row centroids',alpha =0.5)
    plt.plot(t,smooth[:,0],'-',label='Smoothed row centroids')
    plt.plot(t,meds[:,1],'.',label='Col centroids',alpha =0.5)
    plt.plot(t,smooth[:,1],'-',label='Smoothed col centroids')
    #plt.plot(thing,'+')
    plt.ylabel('Displacement (pix)',fontsize=15)
    plt.xlabel('Cadence number',fontsize=15)
    plt.legend(fontsize=12)
    plt.gca().tick_params(which='both',direction='in')
    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=12)
    if type(Save) == str:
        plt.savefig(Save)

def Centroids_DAO(Flux,Median,TPF=None,Plot = False):
    """
    Calculate the centroid shifts of time series images.
    """
    m = Median.copy()
    f = Flux#TPF.flux.copy()
    mean, med, std = sigma_clipped_stats(m, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    s = daofind(m - med)
    mx = s['xcentroid']
    my = s['ycentroid']
    shifts = np.zeros((len(TPF.flux),2,len(mx))) * np.nan
    for i in range(len(f)):
        if np.nansum(f[i]) > 0:
            mean, med, std = sigma_clipped_stats(f[i], sigma=3.0)
            s = daofind(f[i] - med)
            x = s['xcentroid']
            y = s['ycentroid']
            dist = np.zeros((len(mx),len(x)))
            dist = dist + np.sqrt((x[np.newaxis,:] - mx[:,np.newaxis])**2 + (y[np.newaxis,:] - my[:,np.newaxis])**2)

            ind = np.argmin(dist,axis=1)
            indo = np.nanmin(dist) < 1
            ind = ind[indo]
            shifts[i,0,indo] = x[ind] - mx[indo]
            shifts[i,1,indo] = y[ind] - my[indo]


    meds = np.nanmedian(shifts,axis = 2)
    smooth = Smooth_motion(meds,TPF)
    nans = np.nansum(f,axis=(1,2)) ==0
    smooth[nans] = np.nan
    if Plot:
        Plot_centroids(TPF,meds,smooth)
    return smooth


def Shift_images(Offset,Data,median=False):
	"""
	Shifts data by the values given in offset. Breaks horribly if data is all 0.

	"""
	shifted = Data.copy()
	data = Data.copy()
	data[data<0] = 0
	for i in range(len(data)):
		if np.nansum(data[i]) > 0:
			shifted[i] = shift(data[i],[-Offset[i,1],-Offset[i,0]],mode='nearest',order=3)
	return shifted

def Shift_median(Offset,Med):
    """
    Shifts data by the values given in offset. Breaks horribly if data is all 0.

    """
    Offset[np.isnan(Offset)] = 0
    shifted = np.zeros((len(Offset),Med.shape[0],Med.shape[1]))
    Med[np.isnan(Med)] = 0
    #Med[Med<0] = 0
    for i in range(len(Offset)):
        shifted[i] = shift(Med,[Offset[i,1],Offset[i,0]],mode='nearest',order=3)

    return shifted





import scipy
def sgolay2d ( z, window_size, order, derivative=None):
    """
    taken from https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0
    
    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')
    
    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2
    
    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]
    
    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])
        
    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band ) 
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z
    
    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band ) 
    
    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band ) 
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band ) 
    
    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')        
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')        
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')  