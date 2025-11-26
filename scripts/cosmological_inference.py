from .CONFIG_cosmological import *
from time import time


def main_cosmological(data_to_use="Union2.1"):

    likelihood, d, neg_a_mag, arguments, x, X, s, initial_pos, _ = cosmological_likelihood(data_to_use=data_to_use)

    # For plotting purposes:
    z_p, mu_p, _ = read_data_pantheon()
    z_u, mu_u, _ = read_data_union()
    z_d, mu_d, _ = read_data_des()
    # plot_charm1_in_comparison_fields(x_max_pn=np.log(1+np.max(z_p)), x_max_union=np.log(1+np.max(z_u)), show=True)

    print(f"\nUsing {data_to_use} data.\n")

    # global_iterations = 32
    global_iterations = 50

    def new_kl_rate(iter):
        if iter < 25:
            return 7
        elif iter < 48:
            return 20
        else:
            return 100

    inference_start = time()

    posterior_samples = ift.optimize_kl(likelihood_energy=likelihood,
                                        total_iterations=global_iterations,
                                        n_samples=new_kl_rate,
                                        kl_minimizer=descent_finder,
                                        sampling_iteration_controller=ic_sampling_lin,
                                        nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                                        return_final_position=False,
                                        resume=False,
                                        output_directory=f"data_storage/pickled_inferences/cache_{data_to_use}",
                                        initial_position=initial_pos,
                                        plot_energy_history=True
                                        )

    inference_end = time()
    inference_duration = inference_end - inference_start

    now = get_datetime()
    # Save the inference run
    pickle_me_this(f"real/{now}", posterior_samples)

    s_mean, s_var = posterior_samples.sample_stat(s)
    s_err = np.sqrt(X.adjoint(s_var).val)

    current_expansion_mean, current_expansion_err = current_expansion_rate(X.adjoint(s_mean).val, s_err)
    H0_estimate = f"Calculated value of H0: {current_expansion_mean} ± {current_expansion_err}"

    store_meta_data(name=now, duration_of_inference=inference_duration, len_d=len(d.val), expansion_rate=H0_estimate,
                    inference_type='real', signal_model_param=arguments, global_kl_iterations=global_iterations,
                    data_storage_dir_name=f"cache_{data_to_use}")

    # Signal Space Comparison Visualization
    plot_charm2_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1+z_u)),
                                     x_max_des=np.max(np.log(1+z_d)), show=True, save=False, x=x.field().val,
                                     s=X.adjoint(s_mean).val, s_err=s_err, dataset_used=data_to_use,
                                     neg_a_mag=neg_a_mag, )
