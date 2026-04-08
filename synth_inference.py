from charm2 import *
import nifty.cl as ift
import numpy as np

seed = 42
ift.random.push_sseq_from_seed(seed)
np.random.seed(seed)

b0=0.2
# data_args = DataArgs(use_des_like_data_distribution=False, noise_covariance=None, uniform_drawing=False)
data_args = DataArgs(use_des_like_data_distribution=True, noise_covariance=None, uniform_drawing=False,
                     custom_data_shift=0.1, apply_shift_where=("<", 0.6))

# ground_truth_args = GroundTruthArgs(mode="flat_LCDM", H0=100, Ωm0=0.3,)  # synthetic reconstructions from sect. 4.2
ground_truth_args = GroundTruthArgs(mode="flat_LCDM", H0=70.3, Ωm0=0.3,)  #  for bump experiment

LH = synthetic_likelihood(init_fluctuations_parameter=b0, mode='non-parametric',
                          ground_truth_args=ground_truth_args, data_generation_args=data_args,)

global_iterations = 1
# global_iterations = 50
kl_rate = lambda itr: 1
# kl_rate = lambda itr: 25 if itr < 45 else 50

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


posterior_samples = optimize_kl_and_store_metadata(LH, calculate_elbo=False, custom_folder_name="test", **inference_args)

plot_charm2(posterior_samples, LH, plot_domain="signal", plot_mode='synthetic', **dict(show_comparison_fields=True))
