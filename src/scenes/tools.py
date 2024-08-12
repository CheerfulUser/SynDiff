import pandas as pd
import numpy as np
from scipy.optimize import minimize 


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

def mask_rad_func(x, a=7.47132813e+03, b=5.14848986e-01, c=3.15022393e+01):
    rad = np.array(a * np.exp(-b * x) + c)
    rad[rad>300] = 300
    rad = rad.astype(int)
    return rad