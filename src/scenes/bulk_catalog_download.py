import pandas as pd 
from tools import _get_tyco, _get_gaia
from astropy.coordinates import SkyCoord, Angle



sc = SkyCoord(self.ps1.ra,self.ps1.dec, frame='icrs', unit='deg')
gaia = 
bsc = _get_bsc(sc,0.41)