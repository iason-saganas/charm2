from setup_cosmological import *

# likelihood_energy_union = ift.GaussianEnergy(d_u, N_u.inverse) @ R_u  # Union2.1 likelihood
likelihood_energy_pantheon = ift.GaussianEnergy(d_p, N_p.inverse) @ R_p  # Pantheon+ likelihood
global_iterations = 3

posterior_samples = ift.optimize_kl(likelihood_energy=likelihood_energy_pantheon,
                                    total_iterations=global_iterations,
                                    n_samples=kl_sampling_rate,
                                    kl_minimizer=descent_finder,
                                    sampling_iteration_controller=ic_sampling_lin,
                                    nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                                    return_final_position=False,
                                    resume=True,
                                    output_directory=f"output_helper")
# Save the inference run
pickle_me_this(f"real/pantheon+_{arguments}", posterior_samples)
print("Saved posterior samples as pickled object. Parameters of the inference: \n",
      arguments,
      "\nUse `analyze_stored_data` to visualize results.")

s_mean, s_var = posterior_samples.sample_stat(s)

# Signal Space Comparison Visualization
plot_charm2_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1+z_u)), show=True,
                                 save=True, x=x.field().val, s=X.adjoint(s_mean).val,
                                 s_err=np.sqrt(X.adjoint(s_var).val))

