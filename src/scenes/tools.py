import pandas as pd
import numpy as np
from scipy.optimize import minimize 
from copy import deepcopy
from glob import glob
import os

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

def catmag_to_imscale(flux,header):
    a = 2.5/np.log(10)
    tmp = (flux - header['boffset']) / (header['bsoften']*2)
    tmp = np.arcsinh(tmp)*a
    return tmp

def download_skycells(names,path,filters=['r','i','z','y'],overwrite=False,mask=False):
    for name in names:
        for band in filters:
            if mask:
                filename = f'rings.v3.{name}.stk.{band}.unconv.mask.fits'
            else:
                filename = f'rings.v3.{name}.stk.{band}.unconv.fits'
            exist = glob(path + filename)
            if (len(exist) == 0) | overwrite:
                _,projection,cell = name.split('.')
                base = 'wget http://ps1images.stsci.edu//rings.v3.skycell/'
                call = base + f'{projection}/{cell}/{filename} -P {path}'
                os.system(call)
            else:
                pass
                #print(f'{filename} already exists.')


def isolate_stars(cat,only_stars=False,Qf_lim=0.85,psfkron_diff=0.05):
    qf_ind = ((cat.gQfPerfect.values > Qf_lim) & (cat.rQfPerfect.values > Qf_lim) & 
              (cat.iQfPerfect.values > Qf_lim) & (cat.zQfPerfect.values > Qf_lim))
    kron_ind = (cat.rMeanPSFMag.values - cat.rMeanKronMag.values) < psfkron_diff
    ind = qf_ind & kron_ind
    if only_stars:
        cat = cat.iloc[ind]
        cat.loc[:,'star'] = 1
    else:
        cat.loc[:,'star'] = 0
        cat.loc[ind,'star'] = 1
    return cat 

def cut_bad_detections(cat):
    ind = (cat.rMeanPSFMag.values > 0) & (cat.iMeanPSFMag.values > 0) & (cat.zMeanPSFMag.values > 0)
    return cat.iloc[ind]


def query_ps1(ra,dec,radius,only_stars=False,version='dr2'):
    if (version.lower() != 'dr2') & (version.lower() != 'dr1'):
        m = 'Version must be dr2, or dr1'
        raise ValueError(m)
    
    str = f'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/{version.lower()}/mean?ra={ra}&dec={dec}&radius={radius}&nDetections.gte=5&pagesize=-1&format=csv'
    try:
        cat = pd.read_csv(str)
    except pd.errors.EmptyDataError:
        print('No detections')
        cat = []
    cat = isolate_stars(cat,only_stars=only_stars)
    cat = cut_bad_detections(cat)
    return cat 


def ps_psf(params,x,y):
    sdx,sdy,sdxy,k = params
    z = x**2/(2*sdx**2) + y**2/(2*sdy**2) + sdxy*x*y
    #psf = (1 + k*z + z*2.25)**(-1)
    psf = (1 + z + z**2/2 + z**3/6)**(-1) # PGAUSS
    psf /= np.nansum(psf)
    return psf

def psf_minimizer(x0,cut,x,y):
    c = deepcopy(cut)
    #y, x = np.mgrid[:c.shape[0], :c.shape[1]]
    #x = x - c.shape[1] / 2 #+ x0[-2]
    #y = y - c.shape[0] / 2 #+ x0[-1]
    x = x - x0[-2]
    y = y - x0[-1]
    psf = ps_psf(x0[:-2],x,y)
    c /= np.nansum(c)
    res = np.nansum((c-psf)**2)
    return res


def psf_phot(f,cut,psf):
    res = np.nansum((cut-psf*f)**2)
    return res

def mask_rad_func(x, a=2.25512307e+03, b=3.30876377e-01, c=2.70001850e+01):
    rad = np.array(a * np.exp(-b * x) + c)
    rad = rad.astype(int)
    return rad


def _get_gaia(skycoord,rad):

    result = Vizier.query_region(skycoord, catalog=['I/355/gaiadr3'],
                                 radius=Angle(rad, "deg"),column_filters={'Gmag':'<14'})
    if result is None:
        result = None
    else:
        result = result['I/355/gaiadr3'].to_pandas()
    return result

def _get_tyco(skycoord,rad):

    result = Vizier.query_region(skycoord, catalog=['J/PASP/120/1128/catalog'],
                                 radius=Angle(rad, "deg"))
    if result is None:
        result = None
    else:
        result = result['J/PASP/120/1128/catalog'].to_pandas()
        result = result.rename(columns={'RAJ2000':'RA_ICRS','DEJ2000':'DE_ICRS'})   
    return result

def _get_bsc(skycoord,rad):
    result = Vizier.query_region(skycoord, catalog=['V/50/catalog'],
                                 radius=Angle(rad, "deg"),column_filters={'Gmag':'<14'})
    if result is None:
        result = None
    else:
        try:
            result = result['V/50/catalog'].to_pandas()
            result = result.rename(columns={'RAJ2000':'RA_ICRS','DEJ2000':'DE_ICRS'})   
            c = SkyCoord(result.RA_ICRS.values,result.DE_ICRS.values, frame='icrs',unit=(u.hourangle, u.deg))
            result['RA_ICRS'] = c.ra.deg; result['DE_ICRS'] = c.dec.deg
        except:
            result = None
    return result
    
def find_unique2ps1(ps1_cat, viz_cat):
    radius_threshold = 2*u.arcsec
    
    coords_obs = SkyCoord(ra=ps1_cat.raMean, dec=ps1_cat.decMean, unit='deg')
    coords_viz = SkyCoord(ra=viz_cat['RA_ICRS'], dec=viz_cat['DE_ICRS'], unit='deg')
    idx, d2d, d3d = coords_viz.match_to_catalog_3d(coords_obs)
    sep_constraint = d2d >= radius_threshold
    viz_unique = viz_cat[sep_constraint]
 
    return viz_unique

def find_unique2viz(v1, v2):
    radius_threshold = 2*u.arcsec
    
    coords_v1 = SkyCoord(ra=v1['RA_ICRS'], dec=v1['DE_ICRS'], unit='deg')
    coords_v2 = SkyCoord(ra=v2['RA_ICRS'], dec=v2['DE_ICRS'], unit='deg')
    idx, d2d, d3d = coords_v2.match_to_catalog_3d(coords_v1)
    sep_constraint = d2d >= radius_threshold
    v2_unique = v2[sep_constraint]
 
    return v2_unique

def combine_close_sources(cat,distance=100):
    x = cat.x.values; y = cat.y.values
    dist = np.sqrt((x[:,np.newaxis] - x[np.newaxis,:])**2 + (y[:,np.newaxis] - y[np.newaxis,:])**2)
    far = dist > 100
    f = 10**((cat.mag.values-25)/-2.5)
    f = np.tile(f, (len(f), 1))
    f[far] = 0
    fn = np.nansum(f,axis=1)
    mn = -2.5*np.log10(fn) + 25
    cat['mag'] = mn
    return cat


import mastcasjobs

def search_refcat(ra, dec, search_size):
    query = """select n.distance as dstDegrees, r.*
               from fGetNearbyObjEq("""+str(ra)+','+str(dec)+","+str(search_size/2)+""") as n
               inner join refcat2 as r on n.objid=r.objid
               order by n.distance
    """

    jobs = mastcasjobs.MastCasJobs(context="HLSP_ATLAS_REFCAT2")
    results = jobs.quick(query, task_name="python atlas cone search")
    try:
        results = results.to_pandas()
        return(results)
    except:
        return None

def mag2flux(mag,zp):
    f = 10**(2/5*(zp-mag))
    return f

def PS1_to_TESS_mag(PS1):
    zp = 25
    g = mag2flux(PS1.gMeanPSFMag.values,zp)
    r = mag2flux(PS1.rMeanPSFMag.values,zp)
    i = mag2flux(PS1.iMeanPSFMag.values,zp)
    z = mag2flux(PS1.zMeanPSFMag.values,zp)
    y = mag2flux(PS1.yMeanPSFMag.values,zp)

    cr = 0.203; ci = 0.501; cz = 0.108
    cy = 0.186; p = -0.005

    t = (cr*r + ci*i + cz*z + cy*y)*(g/i)**p
    t = -2.5*np.log10(t) + zp
    PS1['tmag'] = t
    return PS1