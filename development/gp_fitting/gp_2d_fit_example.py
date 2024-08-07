import lightkurve lk
import george
import numpy as np
import matplotlib.pyplot as plt

search_result = lk.search_tesscut('ZTF18adaifep')
tpf = search_result.download(cutout_size=90)

res = tr.Quick_reduce(tpf, calibrate=False)

flux = res['flux']
med_frame = tr.Get_ref(flux)
tab = tr.Unified_catalog(tpf, magnitude_limit=18)

col = tab.col.values + 0.5
row = tab.row.values + 0.5
pos = np.array([col, row]).T
index, med_cut, stamps = tr.Isolated_stars(pos, tab['tmag'].values, flux, med_frame, Distance=5, Aperture=5)

test_stamp = stamps[0][100]
print(stamps.shape)

# GP interpolation
pred_stamp = gp_2d_fit(test_stamp, kernel='matern32')

# plot prediction
fig = plt.subplot(1, 2, 1)

plt.imshow(test_stamp, cmap='viridis', aspect=1, interpolation='nearest',  origin='lower')
plt.title('Original Stamp')
plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)

plt.subplot(1 ,2, 2)
plt.imshow(pred_stamp, cmap='viridis', aspect=1, interpolation='nearest', origin='lower')
plt.title('GP prediction')
plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
plt.show()
