from charm2 import *
import nifty.cl as ift
import numpy as np
import argparse

seed = 42
ift.random.push_sseq_from_seed(seed)
np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Run simulation with varying alpha parameter.")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha parameter (e.g., 0.1, 0.5, 1.0)")

    args = parser.parse_args()
    alpha = args.alpha

    survey_to_use_cov_of = "DESY5"
    _, _, des_cov = read_data(survey_to_use_cov_of)

    data_args = DataArgs(use_des_like_data_distribution=True, noise_covariance= alpha * des_cov)
    ground_truth_args = GroundTruthArgs(mode="flat_EDE", H0=70, w0=-0.36, wa=-8.8, Ωm0=0.495)

    LH = synthetic_likelihood(init_fluctuations_parameter=None, mode='flat_LCDM',
                              ground_truth_args=ground_truth_args, data_generation_args=data_args,)

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


    note = (f"This run used the noise covariance of the survey <{survey_to_use_cov_of}> as the base covariance.\n "
            f"A reduction factor was multiplied onto the noise covariance: alpha = {alpha}. ")
    posterior_samples = optimize_kl_and_store_metadata(LH, calculate_elbo=False,
                                                       custom_folder_name=f"mock_study_1/alpha_is_{alpha}",
                                                       custom_note=note,
                                                       **inference_args)

    plot_charm2(posterior_samples, LH, plot_domain="signal", plot_mode='synthetic')

if __name__ == "__main__":
    main()