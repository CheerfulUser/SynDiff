import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
#from schwimmbad import MultiPool
from joblib import Parallel, delayed
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.path as pat
from copy import deepcopy
from PS_image_download import *

from astropy.stats import SigmaClip

# from photutils import Background2D

#from test_convolution import *
from utils import *

from scipy.ndimage import  rotate
from astropy.visualization import (SqrtStretch, ImageNormalize)

def Get_TESS_corners(TESS,PS1_wcs):
    y,x = TESS.flux.shape[1:]
    # include the top corners for the last pixels
    x += 1; y += 1

    corners = np.zeros((2,x,y))
    ps_corners = np.zeros((2,x,y))
    x_arr = np.arange(0,x) - 0.5 # offset from the pixel centers 
    y_arr = np.arange(0,y) - 0.5 # offset from the pixel centers 

    for i in range(x):
        for j in range(y):
            corners[:,i,j] = pix2coord(x_arr[i],y_arr[j],TESS.wcs) # why is this offset by 1????
            ps_corners[:,i,j] = coord2pix(corners[0,i,j],corners[1,i,j],PS1_wcs)
            
    return ps_corners
    
    
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
    
def Pix_sum(Square,squares,pspixels,psimage):
    arr = np.zeros_like(squares)
    contained = squares[Square].contains_points(pspixels)
    if contained.any():
        good = pspixels[contained].astype('int')
        summed = np.nansum(psimage[good[:,1],good[:,0]])
        arr[Square] = summed
    return arr


def PS1_tess_comp(ps1):
    cg = 0
    cr = 0.23482658
    ci = 0.35265516
    cz = 0.27569384
    cy = 0.13800082
    cp = 0.00067772
    fit = ((cg*ps1['g'] + cr*ps1['r'] + ci*ps1['i'] + 
            cz*ps1['z'] + cy*ps1['y'])*(ps1['g']/ps1['i'])**cp)
    fit[~np.isfinite(fit)] = 0
    return fit 

def PS1_tess_frac(ps1):
    fr = 0.6767
    fi = 0.9751
    fz = 0.9773
    fy = 0.6725
    
    comp = fr*ps1['r'] + fi*ps1['i'] + fz*ps1['z'] + fy*ps1['y']
    return comp

def test_global():
    print(pspixels)
    print(squares)

def Regrid_PS(PS1, Corners,cores=7):
    dim1, dim2 = Corners.shape[1:]
    dim1 -= 1; dim2 -= 1
    px, py = np.where(PS1)

    squares = np.array(Make_squares(Corners))
    square_num = np.arange(0,len(squares))

    points = np.zeros((len(px),2))
    points[:,0] = px
    points[:,1] = py

    pspixels = Footprint_square(Corners, points)

    psimage = PS1.copy()
    
    values = Parallel(n_jobs=cores)(delayed(Pix_sum)(sq,squares,pspixels,psimage) for sq in square_num)

    PS_scene = np.array(values)
    PS_scene = np.nansum(PS_scene,axis=0)
    PS_scene = PS_scene.astype('float')
    PS_scene = PS_scene.reshape(dim1,dim2)
    return PS_scene


def Get_PS1(RA, DEC,Size, filt='i'):
    '''
    Size limit seems to be around 1000
    '''
    if Size > 30:
        raise ValueError('Thats too big man')
    Scale = 100
    size = Size * Scale#int((Size + 2*np.sqrt(1/2)*Size) * Scale ) # last term is a fudge factor 
    fitsurl = geturl(RA,DEC, size=size, filters=filt, format="fits")
    if len(fitsurl) > 0:
        fh = fits.open(fitsurl[0])
        return fh[0]
    else:
        raise ValueError("No PS1 images at for this coordinate") 
        return 
    
def PS1_images(RA, DEC,Size,filt):
    """
    Grab all PS1 images and make a dictionary, size is in terms of TESS pixels, 100 less than actual.
    """
    images = {}
    for f in filt:
        im = Get_PS1(RA, DEC,Size,f)
        ima = im.data 
        m = -2.5*np.log10(ima) + 25 + 2.5*np.log10(im.header['EXPTIME'])
        flux = 10**(-2/5*(m-25))
        flux[~np.isfinite(flux)] = 0
        #ima[~np.isfinite(ima) | (ima < 0)] = 0
        images[f] = flux
        #images['exp' + f] = im.header['EXPTIME']
    images['tess']  = PS1_tess_comp(images)
        
    images['wcs'] = WCS(im)
    
    return images



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
    