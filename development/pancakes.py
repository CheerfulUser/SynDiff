import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning

import time
from datetime import datetime
from copy import deepcopy
import os
from itertools import compress

from tqdm import tqdm

from mocpy import MOC
# from mocpy import WCS as mocWCS

from billiard.pool import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import warnings # To ignore our problems
warnings.simplefilter('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# import warnings
# import traceback

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     log = file if hasattr(file, 'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback

class Pancakes():
    def __init__(self, file1, savepath = None, num_cores = 1, sector = None, ccd = None,
                 use_multiple_cores_per_task = False, overwrite = True, buffer = None, 
                 priority = 'low'):
        """
        Initializes the Pancakes class
        """

        self.file1 = file1
        self.use_multiple_cores_per_task = use_multiple_cores_per_task
        self.master_skip = False
        self.overwrite = overwrite
        self.priority = priority

        if isinstance(buffer, int):
            self.buffer = buffer
        elif buffer is None:
            self.buffer = 0
        else:
            print('Buffer must be an integer. Setting to 0')
            self.buffer = 0

        if self.priority not in ['low', 'quality', 'high']:
            print('Priority must be low, quality or high. Setting to low')
            self.priority = 'low'

        self.header_dicts = []

        if savepath is None:
            self.savepath = os.getcwd()
        else:
            self.savepath = savepath

        if sector is None:
            self.sector = ''
        elif isinstance(sector, int):
            self.sector = sector
            print(f"Sector: {sector}")
        else:
            print('Sector must be an integer. Making None')

        if ccd is None:
            self.ccd = 1
        elif isinstance(sector, int):
            self.ccd = ccd
            print(f"CCD: {ccd}")
        else:
            print('CCD must be an integer. Making 1')
            self.ccd = 1

        self.skycells_final = []

        skycell_csv = './skycell_coordinates.csv'
        self.skycell_df = pd.read_csv(skycell_csv)

        skycell_wcs_csv = './SkyCells/skycell_wcs.csv'
        self.skycell_wcs_df = pd.read_csv(skycell_wcs_csv)

        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()-2
        else:
            self.num_cores = min([int(multiprocessing.cpu_count()-2), num_cores]) 

        self._image1(self.file1)
        self._check_master_names()
        self._ravelling()
        self._index_master_file()
        self.name_skycells()
        self._skycelling()
        self._dataframe_for_header()

    def _dataframe_for_header(self):
        """
        Creates a Header of telescope and software information
        """

        dict_for_header = {}

        self.date_mod =datetime.now().strftime('%Y-%m-%d')

        cols = ['SECTOR', 'CAMERA', 'CCD', 'TELESCOP', 'INSTRUME']#, 'DATE-MOD', 'SOFTWARE', 'BSCALE', 'BZERO']
        defaults = ['N/A', 1, 1, 'Not specified', 'Not specified']#, '','','1.0','32768.0']

        for c in range(len(cols)):
            col = cols[c]
            try:
                dict_for_header[col] = self.temp_copy[col]
            except:
                dict_for_header[col] = defaults[c]

        dict_for_header['DATE-MOD'] = self.date_mod
        dict_for_header['SOFTWARE'] = 'SynDiff'
        dict_for_header['CREATOR'] = 'PanCAKES'
        dict_for_header['SVERSION'] = 0.1
        dict_for_header['BSCALE'] = 1.0
        dict_for_header['BZERO'] = 32768.0

        self.base_header = fits.Header(dict_for_header)

    def _index_master_file(self):
        """
        Creates an index mapping to understand the ravelling of the TESS pixels to SkyCells
        """

        full_date =datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%m')

        new_fits_header = deepcopy(self.temp_copy)
        
        new_fits_header['DATE-MOD'] = full_date
        file = os.path.join(self.savepath, self.master_index)

        flattened_indices = np.arange(self.data_shape[0] * self.data_shape[1]).reshape(self.data_shape)
        new = np.stack((flattened_indices, self._tx, self._ty), axis=-1)

        im1_header = deepcopy(new_fits_header)
        im1_header['EXTNAME'] = 'Ravelled Pixels'
        im1_header['XTENSION'] = 'RAVEL'
        im2_header = deepcopy(new_fits_header)
        im2_header['EXTNAME'] = 'X Pixels'
        im2_header['XTENSION'] = 'X'
        im3_header = deepcopy(new_fits_header)
        im3_header['EXTNAME'] = 'Y Pixels'
        im3_header['XTENSION'] = 'Y'

        primary_hdu = fits.PrimaryHDU(header=new_fits_header)
        image_hdu1 = fits.ImageHDU(data=deepcopy(np.int64(new[:,:,0])), header=im1_header)
        image_hdu1.scale('int16', bscale=1.0,bzero=32768.0)
        image_hdu2 = fits.ImageHDU(data=deepcopy(np.int64(new[:,:,1])), header=im2_header)
        image_hdu2.scale('int16', bscale=1.0,bzero=32768.0)
        image_hdu3 = fits.ImageHDU(data=deepcopy(np.int64(new[:,:,2])), header=im3_header)
        image_hdu3.scale('int16', bscale=1.0,bzero=32768.0)
        hdul = fits.HDUList([primary_hdu, image_hdu1, image_hdu2, image_hdu3])

        hdul.writeto(file, overwrite=True)

        compress = 'gzip -f ' + file
        os.system(compress)

    def _check_master_names(self):
        """
        Checking if the master file already exists to skip
        """
        
        try:
            camera = self.temp_copy['CAMERA']
            ccd = self.temp_copy['CCD']
            new_ccd = int(int(ccd) + 4*(int(camera) - 1))
        except:
            print('Fail')
            new_ccd = 1
            self.temp_copy['CAMERA'] = new_ccd
            self.temp_copy['CCD'] = new_ccd

        if self.sector != '':
            sector = str(self.sector).zfill(4)
            self.temp_copy['SECTOR'] = sector
            file_name_master = self.temp_copy['TELESCOP'].strip() + '_' + 's'+sector + '_' + str(new_ccd) + '_master_pixels2skycells.fits'
            file_name_index = self.temp_copy['TELESCOP'].strip() + '_' + 's'+sector + '_' + str(new_ccd) + '_master_index.fits'
        else:
            file_name_master = self.temp_copy['TELESCOP'].strip() + '_' + str(new_ccd) + '_master_pixels2skycells.fits'
            file_name_index = self.temp_copy['TELESCOP'].strip() + '_' + str(new_ccd) + '_master_index.fits'

        self.master_file = file_name_master
        self.master_index = file_name_index

        if os.path.exists(os.path.join(self.savepath, file_name_master + '.gz')):
            if self.overwrite == False:
                self.int_skip = True
                print('Master file already exists. Will skip master file creation...')
            else:
                self.int_skip = False
                print('Master file already exists. Will overwrite master file creation...')
        else:
            self.int_skip = False
            print('Master file does not exist. Will create master file...')

    def butter(self, skycell, skycell_index):
        """
        Main SkyCell processing sub-routine function that gets called by the ps1_pixels_to_im1 function
        """

        self.skip = False
        print('Processing:', skycell, skycell_index)
        self.skycell = skycell
        self.skycell_index = skycell_index
        self._filename()

        if self.skip == False:
            self._ps1_image(skycell)
            self._pixel_vertices()
            self._ps1_ravelling()
            self.moccy()
            if self.skip == False:
                pix_obj_ind = np.arange(len(self.enc_pix))
                pix_obj_etp = deepcopy(self.enc_pix)
                pix_obj_pix = deepcopy(self.enc_pix_indices[pix_obj_ind])

                tup = (pix_obj_ind, pix_obj_etp, pix_obj_pix)
                self.initialize_moc_pixel(tup)

    def _image1(self, im1_file):
        """
        Opens the TESS image and extracts the necessary information
        """

        hdul = fits.open(im1_file)
        try:
            self.temp_copy = deepcopy(hdul[1].header)
        except:
            self.temp_copy = deepcopy(hdul[0].header)
        self.super_data = hdul[1].data
        self.data_shape = np.shape(self.super_data)
        self.super_wcs = WCS(hdul[1].header)
        self.im1_poly = self.super_wcs.calc_footprint()
        hdul.close()

        self.ra_centre, self.dec_centre = self.super_wcs.all_pix2world(self.data_shape[1]/2, self.data_shape[0]/2, 0)

    def ps1_wcs_function(self, skycell):
        """
        Grabs and stores the WCS information for the PanSTARRS1 images
        """

        other_temp = self.skycell_wcs_df[self.skycell_wcs_df['NAME'] == skycell].reset_index(drop=True)

        records = other_temp.to_dict(orient='records')
        header_dict = {k: v for d in records for k, v in d.items()}

        temp_head = fits.Header(header_dict)
        self.ps1_data_shape = (other_temp['NAXIS2'].iloc[0], other_temp['NAXIS1'].iloc[0])

        temp_wcs = WCS(temp_head)
            
        self.header_dicts.append(temp_head)
        self.ps1_wcs_master.append(temp_wcs)

    def _ps1_image(self, skycell):
        """
        Grabs the PanSTARRS1 skycells information for extraction
        """

        other_temp = self.skycell_wcs_df[self.skycell_wcs_df['NAME'] == skycell].reset_index(drop=True)
        self.ps1_data_shape = (other_temp['NAXIS2'].iloc[0], other_temp['NAXIS1'].iloc[0])
            
        temp_ind = self.complete_skycells[self.complete_skycells['Name'] == skycell].index[0]
        self.ps1_wcs = self.ps1_wcs_master[temp_ind]

        corns = np.array([[self.buffer, self.buffer], [self.ps1_data_shape[1] - self.buffer, self.buffer], 
                        [self.ps1_data_shape[1] - self.buffer, self.ps1_data_shape[0] - self.buffer], 
                        [self.buffer, self.ps1_data_shape[0] - self.buffer]])

        ra_corners_ps1, dec_corners_ps1 = self.ps1_wcs.all_pix2world(corns[:, 0], corns[:, 1], 0)

        self.min_ps1_ra = np.min(ra_corners_ps1)
        self.max_ps1_ra = np.max(ra_corners_ps1)

        self.min_ps1_dec = np.min(dec_corners_ps1)
        self.max_ps1_dec = np.max(dec_corners_ps1)

        self.ps1_poly = np.column_stack((ra_corners_ps1, dec_corners_ps1))

    def _ravelling(self):
        """
        Unravels the TESS image to get the pixel coordinates
        """

        t_y, t_x = self.data_shape
        self._ty, self._tx = np.mgrid[:t_y, :t_x]

        ty_input = self._ty.ravel()
        tx_input = self._tx.ravel()

        self.tpix_coord_input = np.column_stack([ty_input, tx_input])

        self._x_im1 = self.tpix_coord_input[:,1]
        self._y_im1 = self.tpix_coord_input[:,0]
        self._ra_im1, self._dec_im1 = self.super_wcs.all_pix2world(self._x_im1, self._y_im1, 0)

        self.ravelled_indices = np.array([(i, j) for i in range(self.data_shape[0]) for j in range(self.data_shape[1])])

    def _ps1_ravelling(self):
        """
        PanSTARRS1 image unravelling
        """

        p_y, p_x = self.ps1_data_shape

        py, px = np.mgrid[:p_y, :p_x]
        
        py_input = py.ravel()
        px_input = px.ravel()

        ppix_coord_input = np.column_stack((py_input, px_input))

        x2 = ppix_coord_input[:,1]
        y2 = ppix_coord_input[:,0]
        self._ra2, self._dec2 = self.ps1_wcs.all_pix2world(x2, y2, 0)
        
    def _pixel_vertices(self):
        """
        Grab the pixel vertices for the TESS image
        """

        self.pixel_vertices = []

        _ra, _dec = self.super_wcs.all_pix2world(self.tpix_coord_input[:, 1], self.tpix_coord_input[:, 0], 0)
        mask = (_ra >= self.min_ps1_ra) & (_ra < self.max_ps1_ra) & (_dec >= self.min_ps1_dec) & (_dec < self.max_ps1_dec)
        filtered_indices = np.where(mask)[0]

        pix_centers = np.column_stack((_ra[filtered_indices], _dec[filtered_indices]))
        self.pix_center_ra = _ra[filtered_indices]
        self.pix_center_dec = _dec[filtered_indices]
        self.im1_indices = self.ravelled_indices[filtered_indices]

        for c in self.tpix_coord_input[filtered_indices]:
            x = c[1]
            y = c[0]

            upper_left = (x - 0.5, y - 0.5)
            upper_right = (x + 0.5, y - 0.5)
            lower_right = (x + 0.5, y + 0.5)
            lower_left = (x - 0.5, y + 0.5)

            t_poly = self.super_wcs.all_pix2world([upper_left, upper_right, lower_right, lower_left], 0)
            self.pixel_vertices.append(t_poly)

    def moccy(self):
        """
        Grabbing enclosed PanSTARRS1 pixels
        """

        ps1_skycoord = SkyCoord(self.ps1_poly, unit="deg", frame="icrs")
        ps1_moc = MOC.from_polygon_skycoord(ps1_skycoord, complement=False, max_depth=21)

        ps1_mask = ps1_moc.contains_lonlat(self.pix_center_ra * u.degree, self.pix_center_dec * u.degree)

        self.enc_pix_vertices = list(compress(self.pixel_vertices, ps1_mask))
        self.enc_pix_center_ra = np.asarray(self.pix_center_ra)[ps1_mask]
        self.enc_pix_center_dec = np.asarray(self.pix_center_dec)[ps1_mask]
        self.enc_pix_indices = np.asarray(self.im1_indices)[ps1_mask]

        if len(self.enc_pix_vertices) > 0:
            self.enc_pix = []
            for t in self.enc_pix_vertices:
                pix_skycoord = SkyCoord(t, unit="deg", frame="icrs")
                tess_pix_moc = MOC.from_polygon_skycoord(pix_skycoord, complement=False, max_depth=21)
                self.enc_pix.append(tess_pix_moc)
        else:
            self.skip = True
            # print('Skipping:', self.skycell, self.skycell_index)

    def major_moccy(self):
        """
        TESS image pixels contained in the SkyCells
        """

        im1_skycoord = SkyCoord(self.im1_poly, unit="deg", frame="icrs")
        self.im1_moc = MOC.from_polygon_skycoord(im1_skycoord, complement=False, max_depth=21)

        self.enc_sc_vertices = self.im1_pixel_vertices
        self.enc_sc_center_ra = np.asarray(self.sc_centers[:, 0])
        self.enc_sc_center_dec = np.asarray(self.sc_centers[:, 1])
        self.enc_sc_indices = np.asarray(self.sc_names)

        enc_sc = []
        for i in tqdm(range(len(self.enc_sc_vertices)), desc='Creating MOCs for TESS pixels to SkyCells'):
            sc_skycoord = SkyCoord(self.enc_sc_vertices[i], unit="deg", frame="icrs")
            sc_moc = MOC.from_polygon_skycoord(sc_skycoord, complement=False, max_depth=21)
            enc_sc.append(sc_moc)
        
        self.enc_sc = enc_sc

    def skycell_initialize_moc_pixel(self, pix_obj):
        """
        SkyCell pixel initialization sub-routine for matching TESS pixels to SkyCells
        """

        pix_ras = self._ra_im1
        pix_decs = self._dec_im1
        im1_poly = self.enc_sc_vertices
        
        im1_pix_index, im1_pix_moc, skycell_ind = pix_obj
        tp = im1_poly[im1_pix_index]

        _min_ra, _max_ra = np.min(tp[:, 0]) - 0.5, np.max(tp[:, 0]) + 0.5
        _min_dec, _max_dec = np.min(tp[:, 1]) - 0.5, np.max(tp[:, 1]) + 0.5

        search_indices = np.where((pix_ras >= _min_ra) & 
                                (pix_ras <= _max_ra) & 
                                (pix_decs >= _min_dec) & 
                                (pix_decs <= _max_dec))[0]
    
        enc_ps1_pix_mask = im1_pix_moc.contains_lonlat(pix_ras[search_indices]*u.degree, pix_decs[search_indices]*u.degree)
        ps1_ind = np.arange(len(pix_ras))
        ps1_ind = ps1_ind[search_indices][enc_ps1_pix_mask]

        temp_ra = np.asarray(pix_ras[search_indices])[enc_ps1_pix_mask]
        temp_dec = np.asarray(pix_decs[search_indices])[enc_ps1_pix_mask]

        if len(temp_ra) != 0:
            return (skycell_ind, temp_ra, temp_dec, ps1_ind)

    def im1_to_skycells(self):
        """
        Main sub-routine for processing the TESS pixels to SkyCells
        """

        if self.int_skip == True:
            print('Skipping...')
        elif self.master_skip == True:
            print('Skipping...')
        else:
            print('Starting...')
            self.major_moccy()
            sc_payload = [(i,etp, self.skycells_id[i]) for (i,etp) in enumerate(self.enc_sc)]

            pix_output = []

            for i in tqdm(range(len(sc_payload)), desc='Processing TESS pixels to SkyCells'):
                pix_out = self.skycell_initialize_moc_pixel(sc_payload[i])
                if pix_out is not None:
                    pix_output.append(pix_out)

            fll = self._skycell_arrayify(pix_output)
            
            # return fll
            self._skycell_fitsify(fll)

    def complete_mapping(self):
        """
        Main function to call to process the TESS pixels to SkyCells and SkyCells to PanSTARRS1 pixels
        """

        if self.master_skip == False:
            if self.int_skip == True:
                print('Starting...')
                self.ps1_pixels_to_im1()
            else:
                self.im1_to_skycells()
                print('Starting...')
                self.ps1_pixels_to_im1()
        else:
            print('No skycells found in the image')

    def _skycell_fitsify(self, fll):
        """
        Optimal SkyCell fits file creation
        """

        full_date =datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%m')

        new_fits_header = deepcopy(self.temp_copy)
        
        new_fits_header['DATE-MOD'] = full_date
        file = os.path.join(self.savepath, self.master_file)
        primary_hdu = fits.PrimaryHDU(header=new_fits_header)
        image_hdu1 = fits.ImageHDU(data=deepcopy(np.int16(fll)), header=new_fits_header)
        table = fits.BinTableHDU.from_columns([fits.Column(name='SKYCELL', format='20A', array=np.asarray(self.skycells_list)),
                                               fits.Column(name='SKYCIND', format='K', array=np.asarray(self.skycells_id))])
        hdul = fits.HDUList([primary_hdu, image_hdu1, table])

        hdul.writeto(file, overwrite=True)

        compress = 'gzip -f ' + file
        os.system(compress)

    def _skycell_arrayify(self, pix_output):
        """
        Unravels the TESS pixels to SkyCells output, determines overlaps and puts them in an array
        """

        fll = np.full(self.data_shape, np.nan)
        overlaps = {}  # Dictionary to store lists of values for each index
        ref_skycells = deepcopy(self.complete_skycells)

        for i in tqdm(range(len(pix_output)), desc='Creating overlaps'):
            unravel_index = np.unravel_index(pix_output[i][-1], self.data_shape)
            fll[unravel_index] = pix_output[i][0]
            
            index_tuples = list(zip(*unravel_index))  # This creates a list of (x, y) tuples

            for index in index_tuples:
                if index not in overlaps:
                    overlaps[index] = []
                overlaps[index].append(pix_output[i][0])

        filtered_overlaps = {index: values for index, values in overlaps.items() if len(values) > 1}

        if self.buffer:
            buffer = self.buffer
        else:
            print('No buffer set. Setting to 120')
            buffer = 120

        for index, values in tqdm(filtered_overlaps.items(), desc="Processing overlaps"):
            filtered_ind = ref_skycells.iloc[values].index.to_list()#reset_index(drop=True)

            element = np.asarray(self.ps1_wcs_master)[filtered_ind]
            _x, _y = index[1], index[0]
            _ra, _dec = self.super_wcs.all_pix2world(_x, _y, 0)

            minny = []
            backup = []

            for i in range(len(filtered_ind)):
                e = element[i]
                f_ind = filtered_ind[i]
                _x1, _y1 = e.all_world2pix(_ra, _dec, 0)
                t_df = ref_skycells.iloc[f_ind]

                listy = [_x1, _y1, t_df['NAXIS2'] - 1 - _x1, t_df['NAXIS1'] - 1 - _y1]

                minny_val = np.nanmin(listy)

                if minny_val >= buffer:
                    minny.append([filtered_ind[i], minny_val])
                backup.append([filtered_ind[i], minny_val])

            minny = np.array(minny)
            backup = np.array(backup)

            if (len(minny) == 0) | (self.priority == 'quality'):
                minny = backup
                _minny_ind = np.argmax(minny[:, 1])
                best_skycell = minny[_minny_ind][0]
            elif self.priority == 'low':
                best_skycell = np.nanmin(minny[:, 0]) # Adds a priority to the skycell
            elif self.priority == 'high':
                best_skycell = np.nanmax(minny[:, 0])

            fll[_y, _x] = best_skycell

        fll[np.isnan(fll)] = -1

        return fll

    def initialize_moc_pixel(self, pix_obj):
        """
        Initializes the PanSTARRS1 pixels to TESS pixels matching
        """

        pix_ras = self._ra2
        pix_decs = self._dec2
        poly = self.enc_pix_vertices

        output = []

        pix_indexs, tess_pix_mocs, tess_inds = pix_obj

        if len(pix_indexs) != 0:

            for i in range(len(pix_indexs)):
                pix_index = pix_indexs[i]
                tess_ind = tess_inds[i]
                tess_pix_moc = tess_pix_mocs[i]

                tp = poly[pix_index]

                _min_ra, _max_ra = np.min(tp[:, 0]) - 0.05, np.max(tp[:, 0]) + 0.05
                _min_dec, _max_dec = np.min(tp[:, 1]) - 0.05, np.max(tp[:, 1]) + 0.05

                search_indices = np.where((pix_ras >= _min_ra) &
                                        (pix_ras <= _max_ra) &
                                        (pix_decs >= _min_dec) &
                                        (pix_decs <= _max_dec))[0]

                enc_ps1_pix_mask = tess_pix_moc.contains_lonlat(pix_ras[search_indices] * u.degree, pix_decs[search_indices] * u.degree)
                ps1_ind = np.arange(len(pix_ras))
                ps1_ind = ps1_ind[search_indices][enc_ps1_pix_mask]

                temp_ra = np.asarray(pix_ras[search_indices])[enc_ps1_pix_mask]
                temp_dec = np.asarray(pix_decs[search_indices])[enc_ps1_pix_mask]

                output.append((tess_ind, temp_ra, temp_dec, ps1_ind))

            fll = self._arrayifying(output)
            self._fitsify(fll)
        else:
            pass
    
    def _arrayifying(self, pix_output):
        """
        Converts the PanSTARRS1 pixels to TESS pixels output to an array
        """

        fll = np.full((self.ps1_data_shape[0], self.ps1_data_shape[1], 2), np.nan)

        for i in range(len(pix_output)):
            unravel_index = np.unravel_index(pix_output[i][-1], self.ps1_data_shape)
            length = len(unravel_index[0])
            zeros = np.zeros(length)
            tup = (unravel_index[0], unravel_index[1], zeros)

            fll[tup[0], tup[1], np.int16(tup[2])] = pix_output[i][0][1]
            fll[tup[0], tup[1], np.int16(tup[2]) + 1] = pix_output[i][0][0]
        
        return fll

    def _filename(self):
        """
        Determines filename for the skycell and a temporary header
        """

        temp_fits_header = deepcopy(self.base_header)

        temp_fits_header['SKYCELL'] = self.skycell
        temp_fits_header['MAPIND'] = self.skycell_index
        temp_fits_header['PROJCELL'] = int(self.skycell.split('.')[-2])
        temp_fits_header['SKYCIND'] = int(self.skycell.split('.')[-1])

        try:
            camera = temp_fits_header['CAMERA']
            ccd = temp_fits_header['CCD']
            new_ccd = int(int(ccd) + 4*(int(camera) - 1))
        except:
            print('Fail')
            new_ccd = 1

        if self.sector != '':
            sector = str(self.sector).zfill(4)
            temp_fits_header['SECTOR'] = sector
            file_name = temp_fits_header['TELESCOP'].strip() + '_' + 's'+sector + '_' + str(new_ccd) + '_' + self.skycell + '_' + str(self.skycell_index).zfill(5) + '.fits'
        else:
            file_name = temp_fits_header['TELESCOP'].strip() + '_' + str(new_ccd) + '_' + self.skycell + '_' + str(self.skycell_index).zfill(5) + '.fits'

        if os.path.exists(os.path.join(self.savepath, file_name + '.gz')):
            if self.overwrite == False:
                self.skip = True
            else:
                self.skip = False
        else:
            self.int_skip = False

        self.file_name = file_name
        self.temp_fits_header = temp_fits_header

    def _fitsify(self, fll):
        """
        Turns the PanSTARRS1 pixels to TESS pixels output into a fits file
        """

        file_name = self.file_name
        
        temp_fits_header = self.temp_fits_header

        new_fits_header = fits.Header()

        new_fits_header['SIMPLE'] = 'T'

        new_fits_header += self.header_dicts[self.skycell_index]
        new_fits_header += temp_fits_header
        
        # new_fits_header.update(self.header_dicts[self.skycell_index])

        im1_header = deepcopy(new_fits_header)
        im2_header = deepcopy(new_fits_header)

        fll[:,:,0][np.isnan(fll[:,:,0])] = -1
        fll[:,:,1][np.isnan(fll[:,:,1])] = -1

        new_fits_header = self._remove_naxis_cards(new_fits_header)
        im1_header = self._reorder_header(im1_header)
        im2_header = self._reorder_header(im2_header)

        file = os.path.join(self.savepath, file_name)
        primary_hdu = fits.PrimaryHDU(header=new_fits_header)
        image_hdu1 = fits.ImageHDU(data=deepcopy(np.int64(fll[:,:,0])), header=im1_header)
        image_hdu1.scale('int16', bscale=1.0,bzero=32768.0)
        image_hdu1.header['EXTNAME'] = 'X'
        image_hdu1.header['BSCALE'] = 1.0
        image_hdu1.header['BZERO'] = 32768.0
        image_hdu2 = fits.ImageHDU(data=deepcopy(np.int64(fll[:,:,1])), header=im2_header)
        image_hdu2.scale('int16', bscale=1.0,bzero=32768.0)
        image_hdu2.header['BSCALE'] = 1.0
        image_hdu2.header['BZERO'] = 32768.0
        image_hdu2.header['EXTNAME'] = 'Y'

        hdul = fits.HDUList([primary_hdu, image_hdu1, image_hdu2])
        hdul.verify('fix')

        hdul.writeto(file, overwrite=True)

        compress = 'gzip -f ' + file
        os.system(compress)

    def _reorder_header(self, header):
        """
        Reorders the header to the correct order
        """
        
        new_header = fits.Header() # Create a new header list with correct order
        
        required_cards = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'PCOUNT', 'GCOUNT'] # Example of required order
        
        for card in required_cards:  # Add cards in the correct order
            if card in header:
                new_header.append((card, header[card]))

        for card in header: # Add remaining cards that are not in the required order
            if card not in required_cards:
                new_header.append((card, header[card]))
        
        return new_header

    def _remove_naxis_cards(self, header):
        """
        Removes NAXIS related cards from the header
        """

        new_header = fits.Header() # Create a new header
        
        for card in header.cards: # Add cards that are not 'NAXIS' related
            if not card[0].startswith('NAXIS'):
                new_header.append(card)
        
        return new_header

    def ps1_pixels_to_im1(self):
        """
        Main function to match PanSTARRS1 pixels to TESS pixels
        """

        if self.master_skip == True:
            print('Skipping...')
        else:
            sky_list = self.skycells_list
            tasks = [(sky_list[i], i) for i in range(len(sky_list))]
            # tasks = [('skycell.2246.021', 2), ('skycell.2246.022', 3)]

            with Pool(processes=self.num_cores) as pool:
                result = pool.imap_unordered(self._process_wrapper, tasks)
                for res in result:
                    pass

    def _process_wrapper(self, args):
        """
        Wrapper function for the ps1_pixels_to_im1 function
        """

        filename,index = args
        self.butter(filename, index)

    def angular_distance(self, ra1, dec1, ra2, dec2):
        """
        Angular distance between two points on the sky
        """

        d1 = np.sin(np.radians(dec1)) * np.sin(np.radians(dec2))
        d2 = np.cos(np.radians(dec1)) * np.cos(np.radians(dec2)) * np.cos(np.radians(ra1 - ra2))
        return np.degrees(np.arccos(d1 + d2))

    def max_radius(self, ra1, dec1, ra2, dec2):
        """
        Maximum angular distance between two points on the sky
        """

        return np.nanmax(self.angular_distance(ra1, dec1, ra2, dec2))

    def name_skycells(self):
        """
        Name and find appropriate skycells for the TESS image
        """

        ra_corners = self.im1_poly[:, 0]
        dec_corners = self.im1_poly[:, 1]

        radius_var = self.max_radius(self.ra_centre, self.dec_centre, ra_corners, dec_corners) + np.sqrt(2) * 0.4

        radius = self.angular_distance(self.ra_centre, self.dec_centre, self.skycell_df['RA'].to_numpy(), self.skycell_df['DEC'].to_numpy())

        mask = radius < radius_var

        self.skycells_list = self.skycell_df[mask].Name.to_numpy()
        self.skycells_id = np.arange(len(self.skycells_list))

        self.complete_skycells = deepcopy(self.skycell_df[mask].reset_index(drop=True))
        complete_wcs_skycells = deepcopy(self.skycell_wcs_df[mask].reset_index(drop=True))

        if len(self.skycells_list) < 1:
            print('No skycells found in the image')
            self.master_skip = True
        
        self.complete_skycells['NAXIS1'] = complete_wcs_skycells['NAXIS1']
        self.complete_skycells['NAXIS2'] = complete_wcs_skycells['NAXIS2']

        self.ps1_wcs_master = []

        for i in tqdm(range(len(self.skycells_list)), desc='Getting all WCS'):
            self.ps1_wcs_function(self.skycells_list[i])

    def _skycelling(self):
        """
        Finds the pixel vertices of the PanSTARRS1 skycells
        """

        im1_pixel_vertices = []

        sc_center_ra, sc_center_dec = self.complete_skycells.RA.values, self.complete_skycells.DEC.values

        self.sc_centers = np.column_stack((sc_center_ra, sc_center_dec))
        sc_names = self.complete_skycells['Name'].values 
        self.sc_names = [int(name.split('.')[1] + name.split('.')[2]) for name in sc_names]

        for i in range(len(self.complete_skycells)):

            ra_corners = np.array([self.complete_skycells.iloc[i][f'RA_Corner{j}'] for j in range(1, 5)])
            dec_corners = np.array([self.complete_skycells.iloc[i][f'DEC_Corner{j}'] for j in range(1, 5)])

            corner1 = self.super_wcs.all_world2pix(ra_corners[0], dec_corners[0], 0)
            corner2 = self.super_wcs.all_world2pix(ra_corners[1], dec_corners[1], 0)
            corner3 = self.super_wcs.all_world2pix(ra_corners[2], dec_corners[2], 0)
            corner4 = self.super_wcs.all_world2pix(ra_corners[3], dec_corners[3], 0)

            sc_poly = self.super_wcs.all_pix2world([corner1, corner2, corner3, corner4], 0)
            im1_pixel_vertices.append(sc_poly)

        self.im1_pixel_vertices = im1_pixel_vertices
