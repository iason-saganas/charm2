import matplotlib.pyplot as plt
import numpy as np
from data_storage.style_components.matplotlib_style import *
from utilitites import *
import nifty8 as ift
from setup_cosmological import *

samples = unpickle_me_this("real/pantheon+_cfm_{'offset_mean': 0, 'offset_std': None, 'fluctuations': (1.8, 1.8), 'loglogavgslope': (-4, 1e-16), 'asperity': None, 'flexibility': None}_lm_{'slope': (2, 5), 'intercept': (30, 30)}.pickle")

# plot_flat_lcdm_fields(x_max=np.max(np.log(1 + z_p)), show=True, save=False)  # Creates a plot of the comparison CMB and SN fields
# plot_charm1_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1+z_u)), show=False,
#                                save=True)


# Extract and visualize posterior samples
posterior_realizations_list = samples
s_mean, s_var = posterior_realizations_list.sample_stat(s)

plot_data_comparison = False
if plot_data_comparison:
    x_ext = x
    s_mean = X.adjoint(s_mean)
    s_var = X.adjoint(s_var)
else:
    x_ext = x_ext
    s_mean = s_mean
    s_var = s_var

# Signal Space Comparison Visualization
plot_charm2_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1+z_u)), show=False,
                                 save=True, x=x.field().val, s=X.adjoint(s_mean).val,
                                 s_err=np.sqrt(X.adjoint(s_var).val))

if plot_data_comparison:

    # Reconstruct data from posterior
    reconstructed_data = build_response(x, ift.Field.from_raw(x, s_mean.val), data_space, neg_a_mag)
    data_from_base_lcdm = build_response(x, ift.Field.from_raw(x, base_s), data_space, neg_a_mag)

    print("Chi Squared between data and base lcdm: ", chi_square(mu, data_from_base_lcdm.val))
    print("Chi Squared between data and reconstruction: ", chi_square(mu, reconstructed_data.val))

    # Data Space Comparison Visualization
    plt.plot(neg_a_mag, mu, ".", color="black", label=r"Real $\mu$")
    plt.plot(neg_a_mag, data_from_base_lcdm.val, ".", label="Data from base", color="red")
    create_plot_1(x=neg_a_mag, y=reconstructed_data.val, color=blue, x_label="$x$", y_label=r"Distance modulus $\mu$",
                  title="Data reconstruction from base and reconstruction", legend_label="Data from Reconstruction")

