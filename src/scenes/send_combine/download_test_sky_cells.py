import pandas as pd 
from tools import download_skycells

files = pd.read_csv('../../development/SkyCells/Sector020/skycell_s20_c11.csv')['Name'].values
savepath = '/home/phys/astronomy/zgl12/SkyCells/New_SC/S20_3_3/'

try:
    download_skycells(files,savepath,filters=['r','i','z','y'],overwrite=False,mask=False)
except:
    pass
try:
    download_skycells(files,savepath,filters=['r','i','z','y'],overwrite=False,mask=True)
except:
    pass
