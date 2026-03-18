from charm2 import *
import nifty.cl as ift
ift.random.push_sseq_from_seed(42)

data_args = DataArgs(use_des_like_data_distribution=True)
ground_truth_args = GroundTruthArgs(mode="flat_EDE", H0=70, w0=-0.36, wa=-8.8, Ωm0=0.495)

LH = synthetic_likelihood(init_fluctuations_parameter=0.2, data_generation_args=data_args,
                          ground_truth_args=ground_truth_args, mode='non-parametric')


global_iterations = 1
kl_rate = lambda itr: 1

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


posterior_samples = optimize_kl_and_store_metadata(LH, calculate_elbo=True, **inference_args)

plot_charm2(posterior_samples, LH, plot_domain="signal", plot_mode='synthetic')
