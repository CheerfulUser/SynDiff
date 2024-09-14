import healpy as hp
import numpy as np
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy.io import fits
import time
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pprint
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from astropy.coordinates import spherical_to_cartesian
from reproject import reproject_interp
import sys
from itertools import compress
from pprint import pprint

from mocpy import MOC
from mocpy import WCS as mocWCS

from billiard.pool import Pool
import multiprocessing
from functools import partial
from multiprocessing import SimpleQueue

from astropy.wcs import FITSFixedWarning

import warnings # To ignore our problems
warnings.filterwarnings('ignore', category=FITSFixedWarning)

import pickle

from glob import glob





def compute_coverage(tess_file,ps1_file,buffer=120,verbose=False,cores=20,save_path='.'):
	tess_data = None
	tess_wcs = None
	tess_poly = None
	with fits.open(tess_file) as tess_hdul:
		tess_data = tess_hdul[1].data
		tess_wcs = WCS(tess_hdul[1].header)
		tess_poly = tess_wcs.calc_footprint()

	ps1_data = None
	ps1_wcs = None
	ps1_poly = None
	with fits.open(ps1_file) as ps1_hdul:
		# ps1_hdul.info()
		ps1_data = ps1_hdul[0].data
		ps1_wcs = WCS(ps1_hdul[0].header)
		ps1_corners = np.array([[0,0],[0,ps1_data.shape[0]],[ps1_data.shape[1],ps1_data.shape[0]],[ps1_data.shape[1],0]])
		offset_corners = np.array([[buffer,buffer],[buffer,-buffer],[-buffer,-buffer],[-buffer,buffer]])
		offset_corners += ps1_corners

		ps1_poly = ps1_wcs.calc_footprint()
		offset_poly = ps1_wcs.all_pix2world(offset_corners,0)
		# print(ps1_hdul[0].header)
	if verbose:
		print("TESS dimensions: %s, %s" % np.shape(tess_data))
	# print("\tTESS polygon: %s" % tess_poly)
	if verbose:
		print("\nPS1 SkyCell dimensions: %s, %s" % np.shape(ps1_data))
	# print("\tPS1 SkyCell polygon: %s" % ps1_poly)

	min_ps1_ra = np.min(ps1_poly[:,0])
	max_ps1_ra = np.max(ps1_poly[:,0])

	min_ps1_dec = np.min(ps1_poly[:,1])
	max_ps1_dec = np.max(ps1_poly[:,1])
	if verbose:
		print("****************************")

	ps1_platescale = 0.258 # arcsec/pixel
	ps1_ps_deg = ps1_platescale/3600.
	ps1_pix_area_sq_deg = ps1_ps_deg**2
	if verbose:
		print("\nArea per PS1 pixel: %s deg^2" % ps1_pix_area_sq_deg)

	# super sample PS1 pixels
	PS1_NSIDE=2097152
	ps1_hp_area_sq_deg = hp.nside2pixarea(nside=PS1_NSIDE, degrees=True)
	if verbose:
		print("Area per PS1 NSIDE %s pixel: %s deg^2" % (PS1_NSIDE, ps1_hp_area_sq_deg))
	hp_per_ps1 = ps1_pix_area_sq_deg/ps1_hp_area_sq_deg
	if verbose:
		print("PS1 NSIDE pixels per native PS1 pixel: %s" % hp_per_ps1)

	tess_platescale = 21.0 # arcsec/pixel
	tess_ps_deg = tess_platescale/3600.
	tess_pix_area_sq_deg = tess_ps_deg**2
	if verbose:
		print("\nArea per TESS pixel: %s deg^2" % tess_pix_area_sq_deg)

	# super sample TESS pixels
	TESS_NSIDE = 32768
	tess_hp_pixel_area = hp.nside2pixarea(nside=TESS_NSIDE, degrees=True)
	if verbose:
		print("Area per TESS NSIDE %s pixel: %s deg^2" % (TESS_NSIDE, tess_hp_pixel_area))
	hp_per_tess = tess_pix_area_sq_deg/tess_hp_pixel_area
	if verbose:
		print("TESS NSIDE pixels per native TESS pixel: %s" % hp_per_tess)

	indices_per_tess = tess_pix_area_sq_deg/ps1_hp_area_sq_deg
	if verbose:
		print("\nPS1 NSIDE pixel per TESS pixel: %s" % indices_per_tess)

	t_y, t_x = np.shape(tess_data)
	ty, tx = np.mgrid[:t_y, :t_x]

	ty_input = ty.ravel()
	tx_input = tx.ravel()

	tpix_coord_input = np.asarray([ty_input, tx_input]).T

	ravelled_indices = np.arange(len(tx_input))

	start = time.time()

	tess_pixel_vertices = []

	_ra, _dec = tess_wcs.all_pix2world(tpix_coord_input[:,1], tpix_coord_input[:,0], 0)
	mask = (_ra >= min_ps1_ra) & (_ra < max_ps1_ra) & (_dec >= min_ps1_dec) & (_dec < max_ps1_dec)
	filtered_indices = np.where(mask)[0]

	tess_pix_centers = np.column_stack((_ra[filtered_indices], _dec[filtered_indices]))
	tess_pix_center_ra = _ra[filtered_indices]
	tess_pix_center_dec = _dec[filtered_indices]
	tess_indices = ravelled_indices[filtered_indices]


	for i, c in enumerate(tpix_coord_input[filtered_indices]):

		x = c[1]
		y = c[0]

		upper_left = (x-0.5, y-0.5)
		upper_right = (x+0.5, y-0.5)
		lower_right = (x+0.5, y+0.5)
		lower_left = (x-0.5, y+0.5)

		t_poly = tess_wcs.all_pix2world([upper_left, upper_right, lower_right, lower_left], 0)
		tess_pixel_vertices.append(t_poly)
	if verbose:
		print("Number of TESS pixels: %s" % len(tess_pixel_vertices))
		print('\nTime taken:', time.time() - start)


	# Ravel PS1 pixels from 2D -> 1D
	start = time.time()

	p_y, p_x = np.shape(ps1_data)
	print(np.shape(ps1_data))

	py, px = np.mgrid[:p_y, :p_x]

	py_input = py.ravel()
	px_input = px.ravel()

	ppix_coord_input = np.asarray([py_input, px_input]).T

	x2 = ppix_coord_input[:,1]
	y2 = ppix_coord_input[:,0]
	_ra2, _dec2 = ps1_wcs.all_pix2world(x2, y2, 0)

	print('\nTime taken:', time.time() - start)

	# Get TESS pixels enclosed by the PS1 footprint.

	start = time.time()

	ps1_skycoord = SkyCoord(ps1_poly, unit="deg", frame="icrs")
	ps1_moc = MOC.from_polygon_skycoord(ps1_skycoord, complement=False, max_depth=21)
	off_skycoord = SkyCoord(offset_poly, unit="deg", frame="icrs")
	off_moc = MOC.from_polygon_skycoord(off_skycoord, complement=False, max_depth=21)

	#ps1_mask = ps1_moc.contains_lonlat(tess_pix_center_ra*u.degree, tess_pix_center_dec*u.degree)
	ps1_mask = off_moc.contains_lonlat(tess_pix_center_ra*u.degree, tess_pix_center_dec*u.degree)
	print("Num pix enc: %s" % ps1_mask.sum())

	print(len(tess_pix_center_ra))

	enc_tess_pix_vertices = list(compress(tess_pixel_vertices, ps1_mask))#[ps1_mask]
	enc_tess_pix_center_ra = np.asarray(tess_pix_center_ra)[ps1_mask]
	enc_tess_pix_center_dec = np.asarray(tess_pix_center_dec)[ps1_mask]
	enc_tess_pix_indices = np.asarray(tess_indices)[ps1_mask]

	enc_tess_pix = []
	for t in enc_tess_pix_vertices:
		tess_pix_skycoord = SkyCoord(t, unit="deg", frame="icrs")
		tess_pix_moc = MOC.from_polygon_skycoord(tess_pix_skycoord, complement=False, max_depth=21)
		enc_tess_pix.append(tess_pix_moc)
		
	print('\nTime taken:', time.time() - start)

	print(len(enc_tess_pix_indices), len(enc_tess_pix))
	tess_pix_payload = [(i,etp,enc_tess_pix_indices[i]) for (i,etp) in enumerate(enc_tess_pix)]


	#Nproc=cores#int(multiprocessing.cpu_count()-2) ## I like to reserve 2 CPUs to do other things
	Nproc=int(multiprocessing.cpu_count()-2) ## I like to reserve 2 CPUs to do other things
	print("Num processes: %s" % Nproc)

	pix_output = []
	with Pool(processes=Nproc, initializer=init_pool, initargs=(_ra2, _dec2, enc_tess_pix_vertices)) as pool:
		pix_output = pool.map(initialize_moc_pixel, tess_pix_payload)

	print("\n Length of output: %s" % len(pix_output))

	print('\nTime taken:', time.time() - start)

	fll = np.full(np.shape(ps1_data), np.nan)

	for i in range(len(pix_output)):
		unravel_index = np.unravel_index(pix_output[i][-1], np.shape(ps1_data))
		# break
		fll[unravel_index] = pix_output[i][0]
	savename = save_path + '/' + ps1_file.split('.stk.')[0].split('/')[-1] + '.npy'
	np.save(savename,fll)

def init_pool(ps1_pixel_ras, ps1_pixel_decs, _tess_pixel_vertices): # shared_queue, 
	global pix_ras, pix_decs, tess_poly
 
	pix_ras = ps1_pixel_ras
	pix_decs = ps1_pixel_decs
	tess_poly = _tess_pixel_vertices
 
	print("\nInitialized pool!")



def initialize_moc_pixel(tess_pix_obj):

	global pix_ras, pix_decs, tess_poly
	
	tess_pix_index, tess_pix_moc, tess_ind = tess_pix_obj
	tp = tess_poly[tess_pix_index]

	_min_ra, _max_ra = np.min(tp[:, 0]) - 0.05, np.max(tp[:, 0]) + 0.05
	_min_dec, _max_dec = np.min(tp[:, 1]) - 0.05, np.max(tp[:, 1]) + 0.05

	search_indices = np.where((pix_ras >= _min_ra) & 
							  (pix_ras <= _max_ra) & 
							  (pix_decs >= _min_dec) & 
							  (pix_decs <= _max_dec))[0]
	
	enc_ps1_pix_mask = tess_pix_moc.contains_lonlat(pix_ras[search_indices]*u.degree, pix_decs[search_indices]*u.degree)
	ps1_ind = np.arange(len(pix_ras))
	ps1_ind = ps1_ind[search_indices][enc_ps1_pix_mask]

	temp_ra = np.asarray(pix_ras[search_indices])[enc_ps1_pix_mask]
	temp_dec = np.asarray(pix_decs[search_indices])[enc_ps1_pix_mask]

	return (tess_ind, temp_ra, temp_dec, ps1_ind)

