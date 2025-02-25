import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from glob import glob
from matplotlib.path import Path
import matplotlib.patches as patches
from astropy.modeling.functional_models import Gaussian2D
from scipy.signal import fftconvolve

from astropy.stats import sigma_clipped_stats
import os

from copy import deepcopy

from ps1_data_handler import ps1_data
from tools import download_skycells, catmag_to_imscale


class pad_skycell():
    def __init__(self,ps1,skycells,datapath,pad=500,
                 catalog=None,psf_std=2.5,run=True,download=True,
                 plot=False):
        self.ps1 = ps1
        self.datapath = datapath 
        self.pad = pad
        self.skycells = skycells
        self._check_download = download
        self.ps1.get_catalog(catalog)
        self.psf_std = psf_std
        
        self._load_ps1()
        
        # set by ps1
        self.overlap = 240 * 2 # overlap pixels
    
        if run:
            self.run()
            if plot:
                self.plot_cells()
                self.plot_image()
        
        
    def _load_ps1(self):
        if type(self.ps1) == str:
            self.ps1 = ps1_data(self.ps1,mask=False,toflux=False)
        self.ps1.set_padding(self.pad)
    
    def run(self):
        self._ovsersize_points()
        self._check_skycells()
        self._skycell_poly()
        self._intersecting_points()
        self._pad_sides()
        self._pad_corners()
        self._cat_fill()
        self.padded = self.ps1.padded
        
    
    def _ovsersize_points(self):
        pad = self.pad
        pc_y,pc_x = self.ps1.wcs.array_shape 
        pc_y /= 2; pc_x /= 2

        oversize_points = np.array([[-pad, pc_y], # clockwise from left side
                                    [pc_x, 2*pc_y+pad],
                                    [2*pc_x+pad, pc_y],
                                    [pc_x, -pad]])

        oversize_corners = np.array([[-pad,-pad], # Clockwise from lower left 
                                    [-pad, 2*pc_y+pad],
                                    [2*pc_x+pad, 2*pc_y+pad],
                                    [2*pc_x+pad, -pad]])
        
        oversized_edges = self.ps1.wcs.all_pix2world(oversize_points,0)
        oversized_corners = self.ps1.wcs.all_pix2world(oversize_corners,0)
        self.oversized_edges = oversized_edges
        self.oversized_corners = oversized_corners
        
    def _check_skycells(self):
        f = np.array([int(a.split('.')[1]) for a in self.skycells['Name'].values])
        ind = f == self.ps1.im_skycell
        self.skycells = self.skycells.iloc[ind]
        
        
    
    def _skycell_poly(self):
        skycells = self.skycells
        corner1 = SkyCoord(skycells['RA_Corner1'].values,skycells['DEC_Corner1'].values,unit='deg')
        corner2 = SkyCoord(skycells['RA_Corner2'].values,skycells['DEC_Corner2'].values,unit='deg')
        corner3 = SkyCoord(skycells['RA_Corner3'].values,skycells['DEC_Corner3'].values,unit='deg')
        corner4 = SkyCoord(skycells['RA_Corner4'].values,skycells['DEC_Corner4'].values,unit='deg')

        paths = []
        for i in range(len(skycells)):
            box = np.array([[skycells['RA_Corner1'].values[i],skycells['DEC_Corner1'].values[i]],
                            [skycells['RA_Corner2'].values[i],skycells['DEC_Corner2'].values[i]],
                            [skycells['RA_Corner3'].values[i],skycells['DEC_Corner3'].values[i]],
                            [skycells['RA_Corner4'].values[i],skycells['DEC_Corner4'].values[i]]])
            paths += [Path(box)]
        paths = np.array(paths)
        self.skycell_paths = paths
        
    
    def _intersecting_points(self):
        
        matched = []
        for p in self.skycell_paths:
            matched += [p.contains_points(self.oversized_edges)]
        matched = np.array(matched)
        side_cells = np.where(matched) # 0: cell index; 1: side index
        if len(side_cells[0]) == 0:
            side_cells=([],[])

        matched = []
        for p in self.skycell_paths:
            matched += [p.contains_points(self.oversized_corners)]
        matched = np.array(matched)
        corner_cells = np.where(matched) # 0: cell index; 1: corner index
        if len(corner_cells[0]) == 0:
            corner_cells=([],[])
        
        self.side_cells = side_cells
        self.corner_cells = corner_cells
        if self._check_download:
            download_skycells(self.skycells['Name'].values[side_cells[0]],path=self.datapath,filters=[self.ps1.band])
            download_skycells(self.skycells['Name'].values[corner_cells[0]],path=self.datapath,filters=[self.ps1.band])
        
    
    def _pad_sides(self):
        side_cells = self.side_cells
        pad = self.pad
        overlap = self.overlap
        for i in range(len(side_cells[0])):
            name = self.skycells.iloc[side_cells[0]].Name.values[i]
            side = side_cells[1][i]
            filename = f'rings.v3.{name}.stk.{self.ps1.band}.unconv.fits'
            hdul = fits.open(self.datapath + filename)
            if len(hdul) == 1:
                j = 0 
            else:
                j = 1
            buff = hdul[j].data

            if side == 0:
                self.ps1.padded[pad:-pad,:pad+10] = buff[:,-(pad+overlap):-overlap+10]
            elif side == 1:
                self.ps1.padded[-(pad+10):,pad:-pad] = buff[overlap-10:(pad+overlap),:]
            elif side == 2:
                self.ps1.padded[pad:-pad,-(pad+10):] = buff[:,overlap-10:(pad+overlap)]
            elif side == 3:
                self.ps1.padded[:pad+10,pad:-pad] = buff[-(pad+overlap):-overlap+10,:]
        
    def _pad_corners(self):
        corner_cells = self.corner_cells
        pad = self.pad
        overlap = self.overlap
        for i in range(len(corner_cells[0])):
            name = self.skycells.iloc[corner_cells[0]].Name.values[i]
            corner = corner_cells[1][i]
            filename = f'rings.v3.{name}.stk.{self.ps1.band}.unconv.fits'
            hdul = fits.open(self.datapath + filename)
            if len(hdul) == 1:
                j = 0 
            else:
                j = 1
            buff = hdul[j].data
            if corner == 0:
                self.ps1.padded[:(pad+10),:(pad+10)] = buff[-(pad+overlap):-overlap+10,-(pad+overlap):-overlap+10]
            elif corner == 1:
                self.ps1.padded[-(pad+10):,:(pad+10)] = buff[overlap-10:(pad+overlap),-(pad+overlap):-overlap+10]
            elif corner == 2:
                self.ps1.padded[-(pad+10):,-(pad+10):] = buff[overlap-10:(pad+overlap),overlap-10:(pad+overlap)]
            elif corner == 3:
                self.ps1.padded[:pad+10,-pad-10:] = buff[-(pad+overlap):-overlap+10,overlap-10:(pad+overlap)]
    
    def _cat_fill(self):
        sides = []; corners = []
        if len(self.side_cells[1]) < 4:
            sides = set(np.arange(0,4))
            done = set(self.side_cells[1])
            sides = np.array(list(sides - done))
        if len(self.corner_cells[1]) < 4:
            corners = set(np.arange(0,4))
            done = set(self.corner_cells[1])
            corners = np.array(list(corners - done))
        
        if (len(sides) > 0) & (len(corners) > 0):
            if self.ps1.cat is not None:
                cat = deepcopy(self.ps1.cat)
                x,y = self.ps1.wcs.all_world2pix(cat['raMean'].values,cat['decMean'].values,0)
                x += self.pad; y += self.pad
                cat['x'] = x.astype(int); cat['y'] = y.astype(int)
                ind = (x > 0) & (y > 0) & (x < self.ps1.padded.shape[1]) & (y < self.ps1.padded.shape[0])
                cat = cat.iloc[ind]
                cat_image = np.zeros_like(self.ps1.padded)
                ind_image = self._catpad_index(sides,corners)
                ind = ind_image[cat['y'].values,cat['x'].values] == 1
                cat = cat.iloc[ind]
                #self.ps1.cat = cat
                #cat_image[cat['y'].values,cat['x'].values] = catmag_to_imscale(cat[f'{self.ps1.band}MeanPSFMag'].values,self.ps1.header)
                cat_image[cat['y'].values,cat['x'].values] = 10**((cat[f'{self.ps1.band}MeanPSFMag'].values-self.ps1.zp)/-2.5)
                g = Gaussian2D(x_stddev=self.psf_std,y_stddev=self.psf_std,x_mean=10,y_mean=10)
                y,x = np.mgrid[:21,:21]
                psf = g(x,y)
                psf /= np.nansum(psf)
                cat_image = fftconvolve(cat_image, psf, mode='same')
                cat_image = catmag_to_imscale(cat_image,self.ps1.header)
                self.ps1.cat_image = cat_image
                self.ind_image = ind_image
                self.ps1.padded[ind_image > 0] = cat_image[ind_image > 0] + self.ps1.image_stats[1]
            else:
                print('No catalog provided!')
            
    def _catpad_index(self,sides,corners):
        pad = self.pad
        catpad = np.zeros_like(self.ps1.padded)
        for side in sides:
            if side == 0:
                catpad[pad:-pad,:pad+10] = 1
            elif side == 1:
                catpad[-(pad+10):,pad:-pad] = 1
            elif side == 2:
                catpad[pad:-pad,-(pad+10):] = 1
            elif side == 3:
                catpad[:pad+10,pad:-pad] = 1
        for corner in corners:
            if corner == 0:
                catpad[:(pad+10),:(pad+10)] = 1
            elif corner == 1:
                catpad[-(pad+10):,:(pad+10)] = 1
            elif corner == 2:
                catpad[-(pad+10):,-(pad+10):] = 1
            elif corner == 3:
                catpad[:pad+10,-pad-10:] = 1
            
        return catpad
    
    def plot_cells(self):
        path = Path(self.ps1.wcs.calc_footprint())
        plt.figure()
        colours = ['C0','C1','C2','C3']
        for p,c in zip(self.oversized_corners,colours):
            plt.plot(p[0],p[1],c+'*')
        for p,c in zip(self.oversized_edges,colours):
            plt.plot(p[0],p[1],c+'x')
            
        for a in [path]:
            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
                ]
            a = a.vertices
            b = np.vstack((a,a[0]))
            test = Path(b,codes=codes)
            patch = patches.PathPatch(test, facecolor='C0', lw=2,alpha=1)
            plt.gca().add_patch(patch)

        for a in self.skycell_paths[self.side_cells[0]]:
            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
                ]
            a = a.vertices
            b = np.vstack((a,a[0]))
            test = Path(b,codes=codes)
            patch = patches.PathPatch(test, facecolor='C1', lw=2,alpha=0.5)
            plt.gca().add_patch(patch)

        for a in self.skycell_paths[self.corner_cells[0]]:
            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
                ]
            a = a.vertices
            b = np.vstack((a,a[0]))
            test = Path(b,codes=codes)
            patch = patches.PathPatch(test, facecolor='C2', lw=2,alpha=0.5)
            plt.gca().add_patch(patch)
            
    def plot_image(self):
        pad = self.pad
        pixel_path = np.array([[pad,pad],
                               [self.ps1.data.shape[0]+pad,pad],
                               [self.ps1.data.shape[0]+pad,self.ps1.data.shape[1]+pad],
                               [pad,self.ps1.data.shape[1]+pad]])
        path = Path(pixel_path)
        
        vmin = np.nanpercentile(self.ps1.padded,16)
        vmax = np.nanpercentile(self.ps1.padded,90)
        plt.figure()
        plt.imshow(self.ps1.padded,vmin=vmin,vmax=vmax,origin='lower')
        codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
                ]
        a = path.vertices
        b = np.vstack((a,a[0]))
        test = Path(b,codes=codes)
        patch = patches.PathPatch(test, facecolor='None', lw=2,alpha=0.5)
        plt.gca().add_patch(patch)


    

