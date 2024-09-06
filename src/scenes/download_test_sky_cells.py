import pandas as pd 
import sys 
from tools import download_skycells

files = pd.read_csv('../../development/SkyCells/Sector020/skycell_s20_c1.csv')['Name'].values
savepath = 'save_directory'

try:
    download_skycells(files,savepath,filters=['r','i','z','y'],overwrite=False,mask=False)
except:
    pass
try:
    download_skycells(files,savepath,filters=['r','i','z','y'],overwrite=False,mask=True)
except:
    pass
