import pandas as pd
import numpy as np
import gzip
from scripts.utilitites import read_data_union, read_data_pantheon, read_data_des


def read_data_des_unordered():
    """
    Copied from DES-SN5YR file `SN_only_cosmosis_likelihood.py` from GitHub.
    See comment at the end of the page of `https://github.com/des-science/DES-SN5YR/tree/main/4_DISTANCES_COVMAT`.
    """
    print("\nReading DESY5 data...")
    filename = "data_storage/raw_data/DES-SN5YR_HD.csv"
    data = pd.read_csv(filename, delimiter=",", comment='#')
    # The only columns that we actually need here are the redshift,
    # distance modulus and distance modulus error

    ww = (data['zHD'] > 0.00)
    # use the vpec corrected redshift for zCMB
    zCMB = data['zHD'][ww]
    # distance modulus and relative stat uncertainties
    mu_obs = data['MU'][ww]
    mu_obs_err = data['MUERR_FINAL'][ww]

    filename = "data_storage/raw_data/DES-SN5YR-STAT+SYS.txt.gz"
    # This data file is just the systematic component of the covariance -
    # we also need to add in the statistical error on the magnitudes
    # that we loaded earlier
    with gzip.open(filename, "rt") as f:
        line = f.readline()
        n = int(line)
        cov_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov_mat[i, j] = float(f.readline())

    # print("\nMAXIMUM STATISTICAL ERROR : \n", np.max(mu_obs_err))
    # print("\nMAXIMUM off diag ERROR : \n", np.max(cov_mat))

    # Now add in the statistical error to the diagonal
    for i in range(n):
        cov_mat[i, i] += mu_obs_err[i] ** 2
    f.close()

    cov_mat = cov_mat[ww][:, ww]

    zCMB = zCMB.values
    mu_obs = mu_obs.values

    return zCMB, mu_obs, cov_mat


z_des_unordered, mu_des_unordered, cov_mat_des_unordered = read_data_des_unordered()

z_d, mu_d, cov_mat_d = read_data_des()
z_u, mu_u, cov_mat_u = read_data_union()
z_p, mu_p, cov_mat_p = read_data_pantheon()


print(np.min(cov_mat_u), np.max(cov_mat_u))
print(np.min(cov_mat_p), np.max(cov_mat_p))

import matplotlib.pyplot as plt

plt.matshow(cov_mat_d, vmin=0, vmax=np.max(cov_mat_d)/20, cmap='viridis')
plt.show()

plt.matshow(cov_mat_u, vmin=0, vmax=np.max(cov_mat_u)/20, cmap='viridis')
plt.show()

plt.matshow(cov_mat_p, vmin=0, vmax=np.max(cov_mat_p)/20, cmap='viridis')
plt.show()


