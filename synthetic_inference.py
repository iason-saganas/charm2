from setup_synthetic import *
from setup_cosmological import z_p
from setup_cosmological import mu_u, mu_p

# Construct random ground truth domain field
ground_truth_model = ift.from_random(s_g.domain)
# Construct ground truth field
ground_truth_field = s_g(ground_truth_model)
# Construct synthetic data
d = R_g(ground_truth_model) + N.draw_sample()

# Plot signal field ground truth, as well as data realizations
plot_synthetic_ground_truth(x=x, ground_truth=X.adjoint(ground_truth_field).val, x_max_pn=np.max(np.log(1+z_p)))
plot_synthetic_data(neg_scale_fac_mag=neg_a_mag, data=d.val, x_max_pn=np.max(np.log(1+z_p)),
                    mu_arrays=np.concatenate((mu_p, mu_u)))

likelihood_energy = ift.GaussianEnergy(d, N.inverse) @ R
global_iterations = 6
"""
posterior_samples = ift.optimize_kl(likelihood_energy=likelihood_energy,
                                    total_iterations=global_iterations,
                                    n_samples=kl_sampling_rate,
                                    kl_minimizer=descent_finder,
                                    sampling_iteration_controller=ic_sampling_lin,
                                    nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                                    output_directory=None,
                                    return_final_position=False)

# Save the inference run
pickle_me_this(f"synthetic/{arguments}", posterior_samples)
"""
# Unpickle the last inference run
posterior_samples = unpickle_me_this("synthetic/cfm_{'offset_mean': 0, 'offset_std': None, 'fluctuations': (1.8, 1.8), 'loglogavgslope': (-4, 1e-16), 'asperity': None, 'flexibility': None}_lm_{'slope': (2, 5), 'intercept': (30, 5)}.pickle")

posterior_realizations_list = posterior_samples
s_mean, s_var = posterior_realizations_list.sample_stat(s)

plot_synthetic_ground_truth(x=x, ground_truth=X.adjoint(ground_truth_field).val, x_max_pn=np.max(np.log(1+z_p)),
                            reconstruction=(X.adjoint(s_mean).val, np.sqrt(X.adjoint(s_var).val)))
