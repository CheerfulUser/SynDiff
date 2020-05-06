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


def Delta_kernal(Scene,Image,Size=13):
	Basis, coeff_0 = Delta_basis(Size)
	bds = []
	for i in range(len(coeff_0)):
		bds += [(0,1)]
	res = minimize(optimize_delta, coeff_0, args=(Basis,Scene,Image),
				   bounds=bds,options={'disp': True})
	k = np.nansum(res.x[:,np.newaxis,np.newaxis]*Basis,axis=0)
	return k