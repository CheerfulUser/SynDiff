from process_ps1 import combine_ps1
import time

datapath = '/home/phys/astronomy/zgl12/SkyCells/New_SC/S20_3_3/'
skycells = '/home/phys/astronomy/zgl12/SkyCells/New_SC/skycell_s20_c11.csv'
savepath = '/home/phys/astronomy/zgl12/SkyCells/New_SC/Combined_Data_60/'
catalog_path = '/home/phys/astronomy/zgl12/SynDiff/SynDiff/src/scenes/skycell_csvs/Sector20/Sector20_ccd11'
catalog_path = '/home/phys/astronomy/zgl12/SynDiff/SynDiff/src/scenes/Sector20_ccd11'

start = time.time()
combine_ps1(datapath, skycells, psf_std = 60, combine = [0.238,0.344,0.283,0.135], 
            catalog_path = catalog_path, savepath = savepath, suffix = 'rizy.conv',
            use_mask = True, overwrite = False, pad = 500, verbose = 0, run = True, cores = 70)
end = time.time()
print(f'Process completed in {(end - start)/3600:.2f} hours.')