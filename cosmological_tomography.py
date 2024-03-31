from setup_cosmological import *

# likelihood_energy_union = ift.GaussianEnergy(d_u, N_u.inverse) @ R_u  # Union2.1 likelihood
likelihood_energy_pantheon = ift.GaussianEnergy(d_p, N_p.inverse) @ R_p  # Pantheon+ likelihood
global_iterations = 6

posterior_samples = ift.optimize_kl(likelihood_energy=likelihood_energy_pantheon,
                                    total_iterations=global_iterations,
                                    n_samples=kl_sampling_rate,
                                    kl_minimizer=descent_finder,
                                    sampling_iteration_controller=ic_sampling_lin,
                                    nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                                    return_final_position=False,
                                    resume=True,
                                    output_directory="output_helper")
# Save the inference run
pickle_me_this(f"real/pantheon+_{arguments}", posterior_samples)
print("Saved posterior samples as pickled object. Parameters of the inference: \n",
      arguments,
      "\nUse `analyze_stored_data` to visualize results.")

