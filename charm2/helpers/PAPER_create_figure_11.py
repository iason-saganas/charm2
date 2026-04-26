import os

import numpy as np

from charm2 import *
from matplotlib.gridspec import GridSpec
from pathlib import Path
import nifty.cl as ift
from datetime import datetime

# comparison of ELBOs und χ^2

b0 = 0.6

alpha = [1, 0.5, 0.1, 0.05, 0.01]  # reduction factor of covariance

elbo_y1 = [-894.191, -899.604, -934.77, -975.062, -1280.19]  # flat ΛCDM ELBO
elbo_y1_err = [0.847919, 0.846007, 0.843463, 0.842865, 0.842069]  # flat ΛCDM ELBO error

if b0 == 0.2:
    elbo_y2 = [-907.925, -911.071, -924.789, -934.488, -966.532]  # charm2 npa ELBO
    elbo_y2_err = [3.03564, 3.00387, 7.62482, 9.13314, 28.82]  # charm2 npa ELBO error
elif b0 == 0.6:
    elbo_y2 = [-905.506,  -908.031, -917.679, -922.94, -968.224]  # charm2 npa ELBO
    elbo_y2_err = [1.76812, 2.14652, 10.8143, 8.56144, 101.711]  # charm2 npa ELBO error
else:
    raise ValueError("b0 must be either 0.2 or 0.6")

elbo_y3 = [-895.28, -895.014, -898.007, -899.369, -985.927]  # flat w0waCDM ELBO
elbo_y3_err = [4.06084, 1.25156, 1.23565, 1.24519, 27.2122]  # flat w0waCDM ELBO error

red_chi2_y1 = [0.9692, 0.9744, 1.0111, 1.0544, 1.3862]  # flat ΛCDM red χ^2
red_chi2_y1_err = [0.0009, 0.0009, 0.0009, 0.0009, 0.0009]  # flat ΛCDM red χ^2 error

if b0 == 0.2:
    red_chi2_y2 = [0.9746, 0.9751, 0.9786, 0.9806, 0.9920]  # charm2 npa red χ^2
    red_chi2_y2_err = [0.0033,0.0033,0.0083, 0.0099, 0.0314]  # charm2 npa red χ^2 error
elif b0 == 0.6:
    red_chi2_y2 = [0.9725, 0.9729, 0.9770, 0.9794, 1.0197]  # charm2 npa red χ^2
    red_chi2_y2_err = [0.0019, 0.0023, 0.0118, 0.0093,  0.1107]  # charm2 npa red χ^2 error

    # Check out denser sampling
    # print(np.linspace(0.4, 0.01, 7))
    # plt.plot(alpha, red_chi2_y2)
    # from scipy.interpolate import interp1d
    # interp_chi2 = interp1d(alpha, red_chi2_y2, kind='linear')
    #
    # dense_alpha = np.linspace(0.4, 0.01, 7)
    # red_chi2_y2_dense = interp_chi2(dense_alpha)
    #
    # plt.plot(alpha, red_chi2_y2)
    # plt.plot(dense_alpha, red_chi2_y2_dense, "b.")
    # plt.show()

else:
    raise ValueError("b0 must be either 0.2 or 0.6")

red_chi2_y3 = [0.9657, 0.9652, 0.9652, 0.9652, 1.0551]  #  flat w0waCDM red χ^2
red_chi2_y3_err = [0.0044, 0.0014, 0.0013, 0.0014, 0.0296]  #  flat w0waCDM red χ^2 error

labels = [r'Flat $\Lambda\mathrm{CDM}$ fit', r'Flat $w_0w_a\mathrm{CDM}$ fit', r'$\texttt{charm2}$ reconstruction']


fontsize_of_legend = 25
fontsize_of_labels = 30
fontsize_of_text = 20

default_height = 2/3*8
default_widht = 10
fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=False, figsize=(default_widht, 2 * default_height))
ax1:plt.axis = axs[0]
ax2:plt.axis = axs[1]

ax1.errorbar(alpha, elbo_y1, yerr=elbo_y1_err, color=green, ecolor=green, lw=2, elinewidth=1, label=r'Flat $\Lambda\mathrm{CDM}$ fit')
ax1.errorbar(alpha, elbo_y2, yerr=elbo_y2_err, color=blue, ecolor=blue, lw=2, elinewidth=1, label=r'$\texttt{charm2}$' + f' $b_0={b0}$ '+ 'reconstruction')
ax1.errorbar(alpha, elbo_y3, yerr=elbo_y3_err, color='black', ecolor='black', lw=2, elinewidth=1, label=r'Flat $w_0w_a\mathrm{CDM}$ fit')

ax2.errorbar(alpha, red_chi2_y1, yerr=red_chi2_y1_err, color=green, ecolor=green, lw=2, elinewidth=1)
ax2.errorbar(alpha, red_chi2_y2, yerr=red_chi2_y2_err, color=blue, ecolor=blue, lw=2, elinewidth=1)
ax2.errorbar(alpha, red_chi2_y3, yerr=red_chi2_y3_err, color='black', ecolor='black', lw=2, elinewidth=1)

ax1.legend(fontsize=fontsize_of_legend)

ax2.set_xlabel("Scaling of DESY5 covariance", fontsize=fontsize_of_labels)
ax1.set_ylabel("ELBO", fontsize=fontsize_of_labels)
ax2.set_ylabel(r"Reduced $\chi^2$", fontsize=fontsize_of_labels)

# Add some embellishements
x_txt_1, y_txt_1 = 0.15, 1.3
x_txt_2, y_txt_2 = 0.85, 1.3
ax2.text(x_txt_1, y_txt_1, "High signal-to-noise", fontsize=fontsize_of_text, horizontalalignment='left')
ax2.text(x_txt_2, y_txt_2, "Low signal-to-noise", fontsize=fontsize_of_text, horizontalalignment='right')

y_arr = 1.28
dx_arr = 0.1
dx_shift = -.07
ax2.annotate("", xy=(x_txt_1-0.03, y_arr), xytext=(x_txt_1+dx_arr, y_arr),
            arrowprops=dict(arrowstyle="->", lw=2, mutation_scale=20))
ax2.annotate("", xytext=(x_txt_2-0.03+dx_shift, y_arr), xy=(x_txt_2+dx_arr+dx_shift, y_arr),
            arrowprops=dict(arrowstyle="->", lw=2, mutation_scale=20))

plt.tight_layout()
plt.show()

stop
current_file = Path(__file__)
project_root = current_file.parents[2]
fig_dir = Path(project_root, "figures")
os.makedirs(fig_dir, exist_ok=True)
fn = f"PAPER_ELBOS_and_chi2.pdf"
filename = str(Path(fig_dir, fn))
plt.savefig(filename)