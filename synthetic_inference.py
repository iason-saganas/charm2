import matplotlib.pyplot as plt
import numpy as np

from data_storage.style_components.matplotlib_style import *
from setup import *

# Construct ground truth
ground_truth_cf = ift.from_random(s.domain, random_type='normal', mean=-1, std=.05)  # The ground truth correlated field (MultiField).
ground_truth_signal = s(ground_truth_cf)  # The ground truth signal field by applying the signal model (cf, _OpChain)
# onto the ground truth correlated field position.
ground_truth_power_spectrum = s.power_spectrum.force(ground_truth_cf)
d = R(ground_truth_cf) + N.draw_sample()

# Plot signal field and power spectrum ground truth, as well as data realizations
plot_all_synthetic_pre_kl(x.field().val, ground_truth_signal.val,
                          np.log(ground_truth_power_spectrum.val[1:int(n_pix)]), d.val, neg_a_mag)

likelihood_energy = ift.GaussianEnergy(d, N.inverse) @ R
global_iterations = 15

posterior_samples = ift.optimize_kl(likelihood_energy=likelihood_energy,
                                    total_iterations=global_iterations,
                                    n_samples=kl_sampling_rate,
                                    kl_minimizer=descent_finder,
                                    sampling_iteration_controller=ic_sampling_lin,
                                    nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                                    output_directory=None,
                                    return_final_position=True)

# ToDo: What exactly is 'posterior_realizations_list'? Are these the initial 'bad' samples as well? In that case the
#  quality of the signal mean would be weighed down. Shouldn't just 'last_position_cf'  be used?
posterior_realizations_list, last_position_cf = posterior_samples

# ToDo: Calculate p_s_var from all power spectra (see old Charm2 versions)
s_mean, s_var = posterior_realizations_list.sample_stat(s)
p_s_mean = posterior_realizations_list.average(s.power_spectrum)

plot_all_synthetic_post_kl(x.field().val, s_mean.val, np.sqrt(s_var.val), ground_truth_signal.val, neg_a_mag)
