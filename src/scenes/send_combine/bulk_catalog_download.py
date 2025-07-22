import numpy as np

import pandas as pd 
from tools import _get_tyco, _get_gaia, _ps1_casjobs, _save_space, _check_exists, query_ps1, _get_bsc
from astropy.coordinates import SkyCoord, Angle

from tqdm import tqdm


def bulk_download_catalogs(tess_fields,savepath,overwrite=False,sector=None):
    tess = pd.read_csv(tess_fields)
    if sector is not None:
        # tess = tess.loc[tess.Sector == sector]
        tess = tess.loc[(tess.Sector == sector) & (tess.Camera == 3) & (tess.CCD == 3)]
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
        name = f'Sector{tess.Sector.iloc[i]}_ccd11' # {4*(tess.Camera.iloc[i]-1) + tess.CCD.iloc[i]}
        if _check_exists(sp+name+'_ps1.csv',overwrite):
            print('PS1...')
            if center[i].ra.deg > -30:
                ps1 = query_ps1(center[i].ra.deg,center[i].dec.deg,rads[i])
                ps1.to_csv(sp+name+'_ps1.csv')
            print('Done')
        if _check_exists(sp+name+'_gaia.csv',overwrite):
            print('Gaia...')
            gaia = _get_gaia(center[i],rads[i])
            gaia.to_csv(sp+name+'_gaia.csv',index=False)
            print('Done')
        if _check_exists(sp+name+'_bsc.csv',overwrite):
            print('BSC...')
            bsc = _get_bsc(center[i],rads[i])
            bsc.to_csv(sp+name+'_bsc.csv',index=False)
            print('Done')

def ps1_mass_downloads(tess_fields,savepath,overwrite=False,sector=None):
    sc = pd.read_csv('/home/phys/astronomy/zgl12/SynDiff/SynDiff/development/SkyCells/Sector020/skycell_s20_c11.csv')
    center = SkyCoord(sc.RA.values,sc.DEC.values,unit='deg')
    dists = np.zeros((len(center),4))
    for i in range(3):
        i += 1
        corn = SkyCoord(sc[f'RA_Corner{i}'].values,sc[f'DEC_Corner{i}'].values,unit='deg')
        dists[:,i-1] = center.separation(corn).deg
    rads = np.nanmax(dists,axis=1)
    
    ps1_df = pd.DataFrame()
    gaia_df = pd.DataFrame()
    bsc_df = pd.DataFrame()
    name = f'Sector20_ccd11' # {4*(tess.Camera.iloc[i]-1) + tess.CCD.iloc[i]}
    
    sp = savepath + f'Sector20/'
    _save_space(sp)
    
    for i in tqdm(range(len(rads)), desc = 'Looping'):
        if _check_exists(sp+name+'_ps1.csv',overwrite):
            if center[i].ra.deg > -30:
                ps1 = query_ps1(center[i].ra.deg,center[i].dec.deg,rads[i])
                ps1_df = pd.concat([ps1_df, ps1])
                # ps1.to_csv(sp+name+'_ps1.csv')
                
        if _check_exists(sp+name+'_gaia.csv',overwrite):
            gaia = _get_gaia(center[i],rads[i])
            gaia_df = pd.concat([gaia_df, gaia])
            # gaia.to_csv(sp+name+'_gaia.csv',index=False)
            
        if _check_exists(sp+name+'_bsc.csv',overwrite):
            bsc = _get_bsc(center[i],rads[i])
            # bsc.to_csv(sp+name+'_bsc.csv',index=False)
            bsc_df = pd.concat([bsc_df, bsc])
    
    ps1_df = ps1_df.drop_duplicates()
    gaia_df = gaia_df.drop_duplicates()
    bsc_df = bsc_df.drop_duplicates()
    
    ps1_df.to_csv(sp+name+'_ps1.csv')
    gaia_df.to_csv(sp+name+'_gaia.csv',index=False)
    bsc_df.to_csv(sp+name+'_bsc.csv',index=False)

tess_fields = '/home/phys/astronomy/zgl12/SynDiff/SynDiff/development/TESS_FFI/TESS_FFI_Coordinates.csv'
savepath = '/home/phys/astronomy/zgl12/SynDiff/SynDiff/development/'
# bulk_download_catalogs(tess_fields,savepath,overwrite=False,sector=20)

ps1_mass_downloads(tess_fields,savepath,overwrite=False,sector=20)