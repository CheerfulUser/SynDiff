from scipy.optimize import minimize
from scipy import signal
from astropy.convolution import Gaussian2DKernel
import numpy as np


def Delta_basis(Size = 13):
	kernal = np.zeros((Size,Size))
	x,y = np.where(kernal==0)
	middle = int(len(x)/2)
	basis = []
	for i in range(len(x)):
		b = kernal.copy()
		if (x[i] == x[middle]) & (y[i] == y[middle]):
			b[x[i],y[i]] = 1
		else:
			b[x[i],y[i]] = 1
			b[x[middle],y[middle]] = -1
		basis += [b]
	basis = np.array(basis)
	coeff = np.ones(len(basis))
	return basis, coeff

def optimize_delta(Coeff, Basis, Scene, TESS, Normalise = True, Offset = False):
	Kernal = np.nansum(Coeff[:,np.newaxis,np.newaxis]*Basis,axis=0)
	template = signal.fftconvolve(Scene, Kernal, mode='same')
	if Offset:
		template = template[offset1:int(3*offset1),offset2-1:int(3*offset2-1)]

	if Normalise:
		template = template / np.nanmax(template)
		tess = TESS.copy() / np.nanmax(TESS)
	else:
		template = template 
		tess = TESS.copy()
	
	return np.nansum(abs(tess - template))


def Delta_kernal(Scene,Image,Size=13,Normalise=True):
	Basis, coeff_0 = Delta_basis(Size)
	bds = []
	for i in range(len(coeff_0)):
		bds += [(0,1)]
	res = minimize(optimize_delta, coeff_0, args=(Basis,Scene,Image,Normalise),
				   bounds=bds,options={'disp': True})
	k = np.nansum(res.x[:,np.newaxis,np.newaxis]*Basis,axis=0)
	if (abs(res.x)>1).any():
		return np.inf
	else:
		return k


def Convolve_image(Image,Kernal):
	template = signal.fftconvolve(Image, Kernal, mode='same')
	return template


def Isolated_kernals(Sources,Size=5,Scenes = False,Median = True):
    '''
    Calculate the Delta convolution kernals for isolated sources.
    
    -------
    Inputs-
    -------
        Sources  array   n x 2 array of images. 0 is scene, 1 is observation
        Size     int     Size of the delta kernal
    --------
    Options-
    --------
        Scenes   bool  If True, it uses the provided scene as the template
        Median   bool  If True, returns the median of all delta kernals
        
    -------
    Output-
    -------
        Kernals  array  If Median == True, returns single kernal, if not, 
                        returns n kernal array
        
    '''
    kernals = []
    for i in range(len(Sources)):
        star = Sources[i,1]
        if scenes:
            blank = Sources[i,0]
        else:
            blank = np.zeros_like(test)
            blank[blank.shape[0]//2+1,blank.shape[1]//2+1] = np.nansum(test)
        k = Delta_kernal(blank,test,Size=Size,Normalise=False)
        kernals += [k]
    kernals = np.array(kernals)
    if Median:
        return np.nanmedian(kernals,axis=0)
    else:
        return kernals