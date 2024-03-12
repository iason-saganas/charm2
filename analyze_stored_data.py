import matplotlib.pyplot as plt
import numpy as np
from data_storage.style_components.matplotlib_style import *
from utilitites import *
import nifty8 as ift
from setup_cosmological import *

extended_by_2 = "data_storage/pickled_inferences/real/Union2.1_signal_space_extended_by_2.pickle"
extended_by_3 = "data_storage/pickled_inferences/real/Union2.1_signal_space_extended_by_3_chi_squared_2.2.pickle"
extended_by_6 = "data_storage/pickled_inferences/real/Union2.1_signal_space_extended_by_6_chi_squared_2.0.pickle"
extended_by_1 = "data_storage/pickled_inferences/real/Union2.1_signal_space_extended_by_1.pickle"

extended_by_3_new = "data_storage/pickled_inferences/real/Union2.1_signal_space_extended_by_3_chi_1.8.pickle"
extended_by_6_new = "data_storage/pickled_inferences/real/Union2.1_signal_space_extended_by_6_new_chi_1.8.pickle"
extended_by_12 = "data_storage/pickled_inferences/real/Union2.1_signal_space_extended_by_12.pickle"

global_it_15_extended_by_12 = "data_storage/pickled_inferences/real/Union2.1_signal_space_ext_by_12_global_it_15.pickle"
global_it_15_extended_by_2 = "data_storage/pickled_inferences/real/Union2_1_signal_space_ext_by_2_global_it_15.pickle"
global_it_15_extended_by_3 = "data_storage/pickled_inferences/real/Union2_1_signal_space_ext_by_3_global_it_15.pickle"

samples = unpickle_me_this("data_storage/pickled_inferences/real/Union2.1_data_cf_parameters_dict_values([30, None, (0.6, 0.1), (-4, 0.5), None, None]).pickle")

# Extract and visualize posterior samples
posterior_realizations_list, last_position_cf = samples

s_mean, s_var = posterior_realizations_list.sample_stat(s)
# p_s_mean = posterior_realizations_list.average(s.power_spectrum)

# Compare to base ΛCDM filled with CMB Planck Mission Parameters
base_s, base_s_err = build_lcdm_with_cmb_parameters(x_ext.field().val)

# Reconstruct data from posterior
#reconstructed_data = build_response(x, ift.Field.from_raw(x, X.adjoint(s_mean).val), data_space, neg_a_mag)
#data_from_base_lcdm = build_response(x, ift.Field.from_raw(x, base_s), data_space, neg_a_mag)

# Data Space Comparison Visualization
#plt.plot(neg_a_mag, mu, ".", color="black", label=r"Real $\mu$")
#plt.plot(neg_a_mag, data_from_base_lcdm.val, ".", label="Data from base", color="red")
#create_plot_1(x=neg_a_mag, y=reconstructed_data.val, color=blue, x_label="$x$", y_label=r"Distance modulus $\mu$",
#              title="Data reconstruction from base and reconstruction", legend_label="Data from Reconstruction")

# Signal Space Comparison Visualization
H0_base = current_expansion_rate_experimental(base_s)
H0_reconstructed = current_expansion_rate_experimental(s_mean.val)

plt.errorbar(x_ext.field().val, base_s, base_s_err, color=red, ecolor=light_red,
             label=r"Base $\Lambda\mathrm{CDM}$ cosmology. $H_0 \approx$ "+f"${H0_base}$.")
plt.errorbar(x=x_ext.field().val, y=s_mean.val, yerr=np.sqrt(s_var.val), color=blue, ecolor=light_blue)
create_plot_1(x_ext.field().val, s_mean.val,
              x_label=r'Negative Scale Factor Magnitude $x=-\mathrm{log}(a)=\mathrm{log}(1+z)$',
              y_label='Signal Field $s(x)$', title="Posterior Reconstruction", color=blue,
              legend_label=r"Posterior Reconstruction $H_0 \approx$ " + f"${H0_reconstructed}$.", x_lim=(0, 24),
              y_lim=(-32, 32))

create_plot_1(x=None, y=np.log(p_s_mean.val),
              x_label=r'Logarithm of $k$ modes',
              y_label=r'Logarithm Of Mean Power Spectrum $\mathrm{log}(p_s(k))$', title="Posterior Reconstruction", color=blue,
              legend_label=r"Posterior Reconstruction $H_0 \approx$ " + f"${H0_reconstructed}$.", #x_lim=(0, 100),
              )#y_lim=(-32, 32))

