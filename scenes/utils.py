import numpy as np

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