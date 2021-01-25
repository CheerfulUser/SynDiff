import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path
sys.path.append('../scenes/')
import syndiff as sd
ra =  336#95.4586
dec = 47#-51.2377
size = 20
scene = sd.Gaia_scene(ra,dec,20,14,Plot = True,Save='test.pdf')
