# Syndiff
Project to simulate TESS images for image subtraction and optimal aperature selection. All critical functions are located in /scenes/syndiff.py. All notebooks in /development are messy and are uesed to develop functions that end up in syndiff.py.  

All data that the scripts can be found on cloudstor:
https://cloudstor.aarnet.edu.au/plus/s/ZuQ5BHcQDZmC37v
Save the contents to the repository's location in a 'data' directory for scripts to call it.

To-do:
1) WCS - Look into DIA 
2) Fix NaNs in PS images 
3) Identify sources in PS image (sep?)
4) Work out why the model PSFs don't seem to match up
5) Optimise the fitting 
6) Make PS image scenes faster (rotate to match TESS PA)
7) Get conversion from PS mags to TESS counts 
8) Create background models 
9) Do subtraction

