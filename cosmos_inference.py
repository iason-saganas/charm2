from charm2 import *
import numpy as np
import nifty.cl as ift

seed = 42
ift.random.push_sseq_from_seed(seed)
np.random.seed(seed)

LH = cosmological_likelihood(data_to_use="DESY5", mode='non-parametric', init_fluctuations_parameter=0.9)
# LH = cosmological_likelihood(data_to_use="DESY5", mode='flat_LCDM')

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


posterior_samples = optimize_kl_and_store_metadata(LH, calculate_elbo=False, **inference_args)

plot_charm2(posterior_samples, LH, plot_domain="signal", plot_mode="real")
