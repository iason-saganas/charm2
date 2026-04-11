from charm2 import *
import nifty.cl as ift
import numpy as np

seed = 42
ift.random.push_sseq_from_seed(seed)
np.random.seed(seed)


_, _, des_cov = read_data_des()
alpha = 0.1
b0 = 0.6

data_args = DataArgs(use_des_like_data_distribution=True, noise_covariance=alpha * des_cov, uniform_drawing=False)
ground_truth_args = GroundTruthArgs(mode="flat_EDE", H0=70, Ωm0=0.495, w0=-0.36, wa=-8.8)  #  for bump experiment


LH = synthetic_likelihood(init_fluctuations_parameter=b0, mode='non-parametric',
                          ground_truth_args=ground_truth_args, data_generation_args=data_args,)

global_iterations = 1
kl_rate = 1

inference_args = dict(likelihood_energy=LH.like,
                        total_iterations=global_iterations,
                        n_samples=kl_rate,
                        kl_minimizer=descent_finder,
                        sampling_iteration_controller=ic_sampling_lin,
                        nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                        return_final_position=False,
                        resume=True,
                        initial_position=LH.meta.init_pos,
                        plot_energy_history=True
                            )


posterior_samples = optimize_kl_and_store_metadata(LH, calculate_elbo=False,
                                                   abs_path_to_pkl="inferences/mock_study_1/npa_b0_is_0_6/alpha_is_0.1/synthetic_des_like_drawing_ground_is_flat_EDE_while_s_model_is_non-parametric/",
                                                   **inference_args)

plot_charm2(posterior_samples, LH, plot_domain="signal", plot_mode='synthetic', **dict(show_comparison_fields=True))
