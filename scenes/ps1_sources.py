
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle

def Get_Catalogue(tpf, Catalog = 'ps1'):
	"""
	Get the coordinates and mag of all sources in the field of view from a specified catalogue.


	I/347/gaia2dis   Distances to 1.33 billion stars in Gaia DR2 (Bailer-Jones+, 2018)

	-------
	Inputs-
	-------
		tpf 				class 	target pixel file lightkurve class
		Catalogue 			str 	Permitted options: 'gaia', 'dist', 'ps1'
	
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
	if Catalog == 'gaia':
		catalog = "I/345/gaia2"
	elif Catalog == 'dist':
		catalog = "I/347/gaia2dis"
	elif Catalog == 'ps1':
		catalog = "II/349/ps1"
	else:
		raise ValueError("{} not recognised as a catalog. Available options: 'gaia', 'dist','ps1'")

	result = Vizier.query_region(c1, catalog=[catalog],
								 radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))
	no_targets_found_message = ValueError('Either no sources were found in the query region '
										  'or Vizier is unavailable')
	#too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
	if result is None:
		raise no_targets_found_message
	elif len(result) == 0:
		raise no_targets_found_message
	result = result[catalog].to_pandas()
	
	return result 


def PS1_to_TESS_mag(PS1):
	"""
	https://arxiv.org/pdf/1706.00495.pdf pg.9
	"""
	#coeffs = np.array([0.6767,0.9751,0.9773,0.6725])
	g = PS1.gmag.values
	r = PS1.rmag.values
	i = PS1.imag.values
	#z = PS1.zmag.values
	#y = PS1.ymag.values

	#t = coeffs[0] * r + coeffs[1] * i #+ coeffs[2] * z + coeffs[3] * y
	t = i - 0.00206*(g - i)**3 - 0.02370*(g - i)**2 + 0.00573*(g - i) - 0.3078
	PS1['tmag'] = t
	return PS1


def Get_PS1(tpf, magnitude_limit = 18, Offset = 10):
	"""
	Get the coordinates and mag of all PS1 sources in the field of view.

	-------
	Inputs-
	-------
		tpf 				class 	target pixel file lightkurve class
		magnitude_limit 	float 	cutoff for Gaia sources
		Offset 				int 	offset for the boundary 

	--------
	Outputs-
	--------
		coords 	array	coordinates of sources
		Gmag 	array 	Gmags of sources
	"""
	result =  Get_Catalogue(tpf, Catalog = 'ps1')
	result = result[np.isfinite(result.rmag) & np.isfinite(result.imag)]# & np.isfinite(result.zmag)& np.isfinite(result.ymag)]
	result = PS1_to_TESS_mag(result)
	
	
	result = result[result.tmag < magnitude_limit]
	if len(result) == 0:
		raise no_targets_found_message
	radecs = np.vstack([result['RAJ2000'], result['DEJ2000']]).T
	coords = tpf.wcs.all_world2pix(radecs, 1) ## TODO, is origin supposed to be zero or one?
	Tessmag = result['tmag'].values
	#Jmag = result['Jmag']
	ind = (((coords[:,0] >= -10) & (coords[:,1] >= -10)) & 
		   ((coords[:,0] < (tpf.shape[1] + 10)) & (coords[:,1] < (tpf.shape[2] + 10))))
	coords = coords[ind]
	Tessmag = Tessmag[ind]
	#Jmag = Jmag[ind]
	return coords, Tessmag