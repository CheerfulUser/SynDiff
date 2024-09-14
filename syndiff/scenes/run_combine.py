from process_ps1 import combine_ps1

skycells = '/home/phys/astronomy/rri38/syndiff/SynDiff/development/SkyCells/Sector020/skycell_s20_c1.csv'
datapath = '/home/phys/astronomy/rri38/syndiff/ps1_data'
catalog_path = '/home/phys/astronomy/rri38/syndiff/catalogs/s20ccd1_'

join_ps1(datapath=datapath,skycells=skycells,catalog_path=catpath,savepath=savepath,verbose=1,overwrite=False)