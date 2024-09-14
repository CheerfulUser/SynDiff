import numpy as np

import pandas as pd 
from tools import _get_tyco, _get_gaia, _ps1_casjobs, _save_space, _check_exists
from astropy.coordinates import SkyCoord, Angle


def bulk_download_catalogs(tess_fields,savepath,overwrite=False,sector=None):
    tess = pd.read_csv(tess_fields)
    if sector is not None:
        tess = tess.loc[tess.Sector == sector]
    center = SkyCoord(tess.RA_center.values,tess.DEC_center.values,unit='deg')
    dists = np.zeros((len(center),4))
    for i in range(3):
        i += 1
        corn = SkyCoord(tess[f'RA_corner{i}'].values,tess[f'DEC_corner{i}'].values,unit='deg')
        dists[:,i-1] = center.separation(corn).deg
    rads = np.nanmax(dists,axis=1) + 0.4
    

    for i in range(len(rads)):
        sp = savepath + f'Sector{tess.Sector.iloc[i]}/'
        _save_space(sp)
        name = f'Sector{tess.Sector.iloc[i]}_ccd{4*(tess.Camera.iloc[i]-1) + tess.CCD.iloc[i]}'
        if _check_exists(savepath+savepath+name+'_ps1.csv',overwrite):
            if center[i].ra.deg > -30:
                ps1 = query_ps1(center[i].ra.deg,center[i].dec.deg,rads[i])
                ps1.to_csv(savepath+name+'_ps1.csv')
        if _check_exists(sp+name+'_gaia.csv',overwrite):
            gaia = _get_gaia(center[i],rads[i])
            gaia.to_csv(sp+name+'_gaia.csv',index=False)
        if _check_exists(sp+name+'_bsc.csv',overwrite):
            bsc = _get_bsc(center[i],rads[i])
            bsc.to_csv(sp+name+'_bsc.csv',index=False)
