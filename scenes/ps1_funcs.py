import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from schwimmbad import MultiPool
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.path as pat

from astropy.stats import SigmaClip
# from photutils import Background2D

#from test_convolution import *
from utils import *

from scipy.ndimage import  rotate
from astropy.visualization import (SqrtStretch, ImageNormalize)

def Get_TESS_corners(TESS,PS1_wcs):
    x,y = TESS.flux.shape[1:]
    # include the top corners for the last pixels
    x += 1; y += 1

    corners = np.zeros((2,x,y))
    ps_corners = np.zeros((2,x,y))
    x_arr = np.arange(0,x)
    y_arr = np.arange(0,y)

    for i in range(x):
        for j in range(y):
            corners[:,i,j] = pix2coord(x_arr[i]-0.5,y_arr[j]-0.5,TESS.wcs)
            ps_corners[:,i,j] = coord2pix(corners[0,i,j],corners[1,i,j],PS1_wcs)
            
    return ps_corners

def Get_PS1(RA, DEC,Size, filt='i'):
    '''
    Size limit seems to be around 1000
    '''
    size = Size * 150 # last term is a fudge factor 
    fitsurl = geturl(RA,DEC, size=size, filters=filt, format="fits")
    if len(fitsurl) > 0:
        fh = fits.open(fitsurl[0])
        ps = fh[0].data
        ps_wcs = WCS(fh[0])
        return ps, ps_wcs
    else:
        raise ValueError("No PS1 images at for this coordinate") 
        return 
    
    
def ps2tessCts(ra, dec, size):
    #ps_x = image array in flux vals
    ps_g, wcs = Get_PS1(ra, dec, size, 'g')
    ps_i, wcs = Get_PS1(ra, dec, size, 'i')
    
    limiting_g = np.power(10, -(22 - 25)/2.5)
    limiting_i = np.power(10, -(21.5 - 25)/2.5)
    
    ps_g[ps_g <= 0] = limiting_g
    ps_i[ps_i <= 0] = limiting_i
    
    #Convert image to mags
    ps_g_mag = -2.5*np.log10(ps_g) + 25
    ps_i_mag = -2.5*np.log10(ps_i) + 25
    
    #Add together in accordance to https://arxiv.org/pdf/1706.00495.pdf pg.9
    #Calculate synthetic magnitude for TESS
    syn_tess_mag = (- 0.00206*np.power(ps_g_mag-ps_i_mag, 3) 
                    - 0.02370*np.power(ps_g_mag-ps_i_mag, 2) 
                    + 0.00573*(ps_g_mag-ps_i_mag) 
                    + ps_i_mag - 0.3078)
    #And now the flux
    syn_tess_cts = np.power(10, -(syn_tess_mag -20.44)/2.5)
    #print(np.nanmean(syn_tess_mag))
    
    return syn_tess_cts, wcs

def Make_squares(Corners):
    squares = []
    for n in range(Corners.shape[1]-1):
        for m in range(Corners.shape[2]-1):
            # define the verticies
            square = np.zeros((4,2))
            square[0,:] = [Corners[0,n,m],Corners[1,n,m]]
            square[1,:] = [Corners[0,n+1,m],Corners[1,n+1,m]]
            square[2,:] = [Corners[0,n+1,m+1],Corners[1,n+1,m+1]]
            square[3,:] = [Corners[0,n,m+1],Corners[1,n,m+1]]
            # define the patch
            path = pat.Path(square)
            squares += [path]
    return squares

def Footprint_square(Corners, Points):
    square = np.zeros((4,2))
    square[0,:] = [Corners[0,0,0],Corners[1,0,0]]
    square[1,:] = [Corners[0,-1,0],Corners[1,-1,0]]
    square[2,:] = [Corners[0,-1,-1],Corners[1,-1,-1]]
    square[3,:] = [Corners[0,0,-1],Corners[1,0,-1]]
    path = pat.Path(square)
    contained = path.contains_points(Points)
    points = Points[contained] 
    return points
    
def Pix_sum(Square):
    arr = np.zeros_like(squares)
    contained = squares[Square].contains_points(pspixels)
    if contained.any():
        good = pspixels[contained].astype('int')
        summed = np.nansum(psimage[good[:,1],good[:,0]])
        arr[Square] = summed
    return arr

def Regrid_PS(PS1, Corners):
    dim1, dim2 = Corners.shape[1:]
    dim1 -= 1; dim2 -= 1
    global px, py
    px, py = np.where(PS1)
    global squares
    squares = np.array(Make_squares(Corners))
    square_num = np.arange(0,len(squares))

    points = np.zeros((len(px),2))
    points[:,0] = px
    points[:,1] = py

    global pspixels
    pspixels = Footprint_square(Corners, points)

    global psimage
    psimage = PS1.copy()
    
    pool = MultiPool()
    values = list(pool.map(Pix_sum, square_num))
    pool.close()

    PS_scene = np.array(values)
    PS_scene = np.nansum(PS_scene,axis=0)
    PS_scene = PS_scene.astype('float')
    PS_scene = PS_scene.reshape(dim1,dim2)
    return PS_scene

def Photutils_background(Flux):
    """
    Uses Photutils to estimate the background flux.
    
    Inputs:
    -------
    Flux - 3d array

    Outputs:
    -------
    bkg - background model
    std - rms error of the background model 

    """
    bkg = np.zeros_like(Flux)
    std = np.zeros_like(Flux)
    sigma_clip = SigmaClip(sigma=3.)
    #bkg_estimator = SExtractorBackground()
    beep = Background2D(Flux, (3, 3), sigma_clip=sigma_clip,exclude_percentile=70)
    bkg = beep.background
    std = beep.background_rms
    return bkg, std


def PS1_scene(RA, DEC, Size, Convolve = 'PS1', Figures = False):
    tess = Get_TESS(RA,DEC,Size)
    
    PS1_image, PS1_wcs = ps2tessCts(RA,DEC,Size)
    
    if np.isnan(PS1_image).any():
        #bkg, = Photutils_background(PS1_image)
        print('PS1 image for the region is incomplete.'
               ' NaNs are present in image. NaN values are set to background value')
        PS1_image[np.isnan(PS1_image)] = 0#bkg
    
    if Convolve == 'PS1':
        try:
            kernal = Interp_PRF(tess.row + (Size/2), tess.column + (Size/2),
                                tess.camera, tess.ccd, tess.sector)
            PS1_image = signal.fftconvolve(PS1_image, kernal,mode='same')
        except MemoryError:
            raise MemoryError("The convolution is too large, try a smaller array.")
            
    tess_corners = Get_TESS_corners(tess, PS1_wcs)
    if Figures:
        plt.figure(figsize=(6,3))
        #plt.subplot(projection=ps_wcs)
        plt.subplot(1,2,1)
        plt.title('PS1 image')
        plt.imshow(PS1_image/np.nanmax(PS1_image),origin='lower',vmax=.1,cmap='Greys')
        x, y = tess.flux.shape[1:]
        x += 1; y += 1
        z = np.arange(0,x*y,1)
        plt.scatter(tess_corners[0,:,:].flatten(),tess_corners[1,:,:].flatten(),c=z,s=1)

        plt.subplot(1,2,2)
        plt.title('TESS image')
        plt.imshow(tess.flux[0]/np.nanmax(tess.flux[0]),origin='lower',vmax=.1)
    
    ps1_scene = Regrid_PS(PS1_image,tess_corners)
    
    if Convolve == 'Scene':
        PRF = Get_PRF(tess.row + (Size/2), tess.column + (Size/2),
                      tess.camera,tess.ccd, tess.sector)
        ps1_scene = signal.fftconvolve(ps1_scene, PRF, mode='same')
        
    if Figures:
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.title('PS1 scene')
        plt.imshow(rotate(np.flipud(ps1_scene/np.nanmax(ps1_scene)),-90),origin='lower',vmax=0.1)
        plt.subplot(1,2,2)
        plt.title('TESS image')
        plt.imshow(tess.flux[0]/np.nanmax(tess.flux[0]),origin='lower',vmax=0.1)
    
    return ps1_scene
    