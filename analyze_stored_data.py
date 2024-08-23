import matplotlib.pyplot as plt
import numpy as np
from data_storage.style_components.matplotlib_style import *
from utilitites import *
import nifty8 as ift
from setup_cosmological import *

samples = unpickle_me_this("real/pantheon+_cfm_{'offset_mean': 0, 'offset_std': None, 'fluctuations': (0.06, 0.03), 'loglogavgslope': (-4, 1e-16), 'asperity': None, 'flexibility': None}_lm_{'slope': (2, 5), 'intercept': (30, 5)}.pickle",
                           absolute_path=False)

# plot_flat_lcdm_fields(x_max=np.max(np.log(1 + z_p)), show=True, save=False)  # Creates a plot of the comparison CMB and SN fields
# plot_charm1_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1+z_u)), show=False,
#                                save=True)

# Extract and visualize posterior samples
posterior_realizations_list = samples

# Signal Space Comparison Visualization
s_mean, s_var = posterior_realizations_list.sample_stat(s)
plot_charm2_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1+z_u)), show=True,
                                 save=False, x=x.field().val, s=X.adjoint(s_mean).val,
                                 s_err=np.sqrt(X.adjoint(s_var).val))


plot_data_comparison = False
if plot_data_comparison:
    x_ext = x
    s_mean = X.adjoint(s_mean)
    s_var = X.adjoint(s_var)
else:
    x_ext = x_ext
    s_mean = s_mean
    s_var = s_var


if plot_data_comparison:

    # Reconstruct data from posterior
    print("comparison for PANTHEON Compilation dataset (if not change code here)")
    reconstructed_data = build_response(x, ift.Field.from_raw(x, s_mean.val), data_space_p, neg_a_mag_p)
    s_cmb, s_cmb_err = build_flat_lcdm(x.field().val, mode='CMB')
    s_sn, s_sn_err = build_flat_lcdm(x.field().val, mode='SN')
    data_from_cmb_lcdm = build_response(x, ift.Field.from_raw(x, s_cmb), data_space_p, neg_a_mag_p)
    data_from_sn_lcdm = build_response(x, ift.Field.from_raw(x, s_sn), data_space_p, neg_a_mag_p)

    print("Chi Squared between data and base lcdm CMB: ", chi_square(mu_p, data_from_cmb_lcdm.val))
    print("Chi Squared between data and base lcdm SN: ", chi_square(mu_p, data_from_sn_lcdm.val))
    print("Chi Squared between data and RECONSTRUCTION: ", chi_square(mu_p, reconstructed_data.val))

    # Data Space Comparison Visualization
    plt.plot(neg_a_mag_p, mu_p, ".", color="black", label=r"Real $\mu$")
    plt.plot(neg_a_mag_p, data_from_cmb_lcdm.val, ".", label="Data from base CMB", color="red")
    plt.plot(neg_a_mag_p, data_from_sn_lcdm.val, ".", label="Data from base SN", color="green")
    create_plot_1(x=neg_a_mag_p, y=reconstructed_data.val, color=blue, x_label="$x$", y_label=r"Distance modulus $\mu$",
                  title="Data reconstruction from base and reconstruction", legend_label="Data from Reconstruction")

