from .CONFIG_synthetic import *
from time import time


def main_synthetic(plot_ground_truth, plot_mock_data):

    # For plotting purposes:
    z_p, mu_p, _ = read_data_pantheon()
    z_u, mu_u, _ = read_data_union()

    # Construct random ground truth domain field
    ground_truth_model = ift.from_random(s_g.domain)
    # Construct ground truth field
    ground_truth_field = s_g(ground_truth_model)
    # Construct synthetic data
    d = R_g(ground_truth_model) + N.draw_sample()

    # if wished, add a systematic increase / decrease of SN mags at high redshift

    bump_idx = np.where(neg_a_mag > 0.44)
    bump_vals = np.zeros_like(d.val)
    # bump_vals[bump_idx] = 0.5
    bump_vals[bump_idx] = -0.04
    d = d + ift.Field.from_raw(d.domain, arr=bump_vals)

    # Plot signal field ground truth, as well as data realizations
    if plot_ground_truth:
        plot_synthetic_ground_truth(x=x, ground_truth=X.adjoint(ground_truth_field).val, x_max_pn=np.max(np.log(1+z_p)),
                                    save=False, show=plot_ground_truth)
    if plot_mock_data:
        plot_synthetic_data(neg_scale_fac_mag=neg_a_mag, data=d.val, x_max_pn=np.max(np.log(1+z_p)),
                            mu_array=np.concatenate((mu_p, mu_u)), save=False, show=plot_mock_data)

    likelihood_energy = ift.GaussianEnergy(d, N.inverse) @ R
    global_iterations = 6

    inference_start = time()

    posterior_samples = ift.optimize_kl(likelihood_energy=likelihood_energy,
                                        total_iterations=global_iterations,
                                        n_samples=kl_sampling_rate,
                                        kl_minimizer=descent_finder,
                                        sampling_iteration_controller=ic_sampling_lin,
                                        nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                                        output_directory="data_storage/pickled_inferences/temp",
                                        return_final_position=False,
                                        resume=False)

    inference_end = time()
    inference_duration = inference_end - inference_start

    now = get_datetime()
    # Save the inference run
    pickle_me_this(f"synthetic/{now}", posterior_samples)

    store_meta_data(name=now, duration_of_inference=inference_duration, len_d=len(d.val),
                    inference_type='synthetic', signal_model_param=arguments, global_kl_iterations=global_iterations)

    posterior_realizations_list = posterior_samples
    s_mean, s_var = posterior_realizations_list.sample_stat(s)

    plot_synthetic_ground_truth(x=x, ground_truth=X.adjoint(ground_truth_field).val, x_max_pn=np.max(np.log(1+z_p)),
                                reconstruction=(X.adjoint(s_mean).val, np.sqrt(X.adjoint(s_var).val)), save=True,
                                show=True)
