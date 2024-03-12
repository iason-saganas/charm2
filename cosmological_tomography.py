import matplotlib.pyplot as plt
import numpy as np
from data_storage.style_components.matplotlib_style import *
from setup_cosmological import *

# ToDo: I need to understand the Pantheon+ Data better

likelihood_energy = ift.GaussianEnergy(d, N.inverse) @ R
global_iterations = 6

posterior_samples = ift.optimize_kl(likelihood_energy=likelihood_energy,
                                    total_iterations=global_iterations,
                                    n_samples=kl_sampling_rate,
                                    kl_minimizer=descent_finder,
                                    sampling_iteration_controller=ic_sampling_lin,
                                    nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                                    output_directory='Output',
                                    return_final_position=True,
                                    resume=False)
# Save the inference run
pickle_me_this('Union2.1_data_cf_parameters_'+cf_parameters, posterior_samples)
print('Saved posterior samples as ', 'Union2.1_data_cf_parameters_' + cf_parameters + '.pickle \n.'
      'Use "analyze_stored_data.py" to visualize results.')

'''# Extract and visualize posterior samples
posterior_realizations_list, last_position_cf = posterior_samples

s_mean, s_var = posterior_realizations_list.sample_stat(s)
p_s_mean = posterior_realizations_list.average(s.power_spectrum)

x = x.field().val
base_s, base_s_err = build_lcdm_with_cmb_parameters(x)
H0_base = current_expansion_rate_experimental(base_s)
H0_reconstructed = current_expansion_rate_experimental(s_mean.val)
plt.plot(x, base_s, color=red, label=r"Base $\Lambda\mathrm{CDM}$ cosmology. $H_0 \approx$ "+f"${H0_base}$.")
create_plot_1(x, s_mean.val, x_label=r'Negative Scale Factor Magnitude $x=-\mathrm{log}(a)=\mathrm{log}(1+z)$',
              y_label='Signal Field $s(x)$', title="Posterior Reconstruction", color=blue,
              legend_label=r"Posterior Reconstruction $H_0 \approx$ " + f"${H0_reconstructed}$.")'''

