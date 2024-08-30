from download_ps1_skycell import bulk_download

%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt

ra = 120.434250672726+0.25
dec = 45.293553239629475

bulk_download(ra,dec, 12, home_dir = 'data/star_field/',domask=False)