import matplotlib.pyplot as plt
from data_storage.style_components.matplotlib_style import *
import nifty8 as ift

n_pix = 8000
n_datapoints = 200

# Domain of interest is 0 < x < 1, where x = log(1+z) is called the 'negative scale factor magnitude'.
# Here, we stretch the signal domain length to 6.7, so the periodic boundary conditions set by NIFTy do not affect
# the reconstruction in the domain of interest. 6.7 is log(1+z), where z is approximately the CMB's redshift.
x_length = 6.7

noise_level = .1

x = ift.RGSpace(n_pix, distances=x_length/n_pix)
