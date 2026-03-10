# import nifty8 as ift
import nifty.cl as ift
from scipy.signal.windows import tukey

from .utilitites import *
import numpy as np

from .utilitites import construct_initial_position

# Defining necessary spaces, fields and operators:
# Geometric object 'x' as signal space. The corresponding values (coordinate points) can be grabbed via x.field().val
# Set the resolution of the signal space
# Build the to-be-inferred signal from a correlated field
# Build the signal response as an operator chain '_OpChain'
# Construct synthetic lines of sight
# Define the Covariance of the Gaussian Likelihood as a 'ScalingOperator'
# Define iteration and minimization control

from .custom_correlated_field import CustomSimpleCorrelatedField
from .CONFIG_cosmological import ic_sampling_lin, ic_sampling_nl, geoVI_sampling_minimizer, ic_newton, descent_finder

__all__ = ["synthetic_likelihood"]

def synthetic_likelihood(init_fluctuations_parameter, n_dp=500, use_des_like_data=False, uniform_drawing=False, ground_truth_Omega_m=None):
    """

    :param init_fluctuations_parameter: np.float,   The initial fluctuations parameter
    :param n_dp:                                    # of synthetic data points to draw
    :param use_des_like_data:                       If True, `n_dp` is overriden with 1829 and the data points are distributed similarly
                                                    the the real DESY5 dataset.
    :param ground_truth_Omega_m                     If float, uses this value to construct the ground truth from a flat ΛCDM model.
                                                    Otherwise, defaults to s_SN as ground truth which is used in the paper.
    :return:
    """
    n_dp = n_dp  # 500
    if use_des_like_data:
        des_hist_edges = np.loadtxt(f"{data_dir}/desy5_data_histogram_bin_edges.txt")
        des_hist_counts = np.loadtxt(f"{data_dir}/desy5_data_histogram_bin_counts.txt")

        neg_a_mag = unidirectional_radial_los(1829, uniform_drawing=True, end_of_data=0.7514160887, specific_hist=(des_hist_edges, des_hist_counts))  # The negative scale factor magnitude,
        # x = -log(a) = log(1+z)
    else:
        neg_a_mag = unidirectional_radial_los(n_dp, uniform_drawing=uniform_drawing)

    config = {
        'Signal Field Resolution': 4096,
        'Length of signal space': np.max(neg_a_mag),
        'Fac to extend signal space by': 2,
        'Noise level': .1,
    }

    n_pix, x_length, x_fac, noise_level = [float(setting) for setting in config.values()]

    pxl_size = x_length / n_pix
    x = ift.RGSpace(int(n_pix), distances=pxl_size)  # The Signal space.
    x_ext = ift.RGSpace(int(x_fac*n_pix), distances=pxl_size)  # The extended signal space
    x = attach_custom_field_method(x)  # Attach `field()` method
    x_ext = attach_custom_field_method(x_ext)  # Attach `field()` method
    data_space = ift.UnstructuredDomain((n_dp,))

    # Arguments of the correlated field model
    args_cfm = {
        'offset_mean': 0,
        'offset_std': None,
        'loglogavgslope': (-4, 1e-16),
        'fluctuations': (.2, .14),
        'asperity': None,
        'flexibility': None,
    }

    # Arguments of the line model
    args_lm = {
        'slope': (2, 5),
        'intercept': (30, 10)
    }

    X = ift.FieldZeroPadder(domain=x, new_shape=(x_fac*n_pix, ))

    # The to-be-inferred signal on the extended domain
    cfm = CustomSimpleCorrelatedField(target=x_ext, use_uniform_prior_on_fluctuations=True, tukey_taper_ends=False,
                                      **args_cfm)
    line = LineModel(target=x_ext, args=args_lm)
    s = cfm + line

    # ift.plot_priorsamples(s)
    # The ground truth model
    # s_g = PiecewiseLinear(signal_space=x_ext, omega_m_custom=0.3, omega_l_custom=2, high_curv=True)  # for high curvature choose this and set the matter exponent to 5 instead of 3
    if ground_truth_Omega_m:
        Ωm = ground_truth_Omega_m
        s_g = PiecewiseLinear(signal_space=x_ext, omega_m_custom=Ωm, omega_l_custom=1-Ωm, high_curv=False,
                              offset_custom=29.81)
    else:
        # Standard flat LCDM model with s_sn set as ground truth
        s_g = PiecewiseLinear(signal_space=x_ext, omega_m_custom=0.334, omega_l_custom=0.666, high_curv=False,
                              offset_custom=29.81)

    arguments = 'cfm_' + str(args_cfm) + '_lm_' + str(args_lm)

    # Build the signal response, noise operator, data field and others
    R = build_response(signal_space=x, signal=X.adjoint @ s, data_space=data_space, neg_scale_factor_mag=neg_a_mag)
    R_g = build_response(signal_space=x, signal=X.adjoint @ s_g, data_space=data_space, neg_scale_factor_mag=neg_a_mag)
    # Ground truth response
    N = ift.ScalingOperator(domain=(data_space, ), factor=noise_level, sampling_dtype=np.float64)

    initial_pos = construct_initial_position(n_pix_ext=int(n_pix * x_fac), distances=pxl_size,
                                             fluctuations=init_fluctuations_parameter, apply_prior_xi_s=True)

    # Construct random ground truth domain field
    ground_truth_model = ift.from_random(s_g.domain)

    # Construct ground truth field
    ground_truth_field = s_g(ground_truth_model)
    # Construct synthetic data
    d = R_g(ground_truth_model) + N.draw_sample()

    # DO NOT DELETE
    # If wished, add a systematic increase / decrease of SN mags at high redshift

    # bump_idx = np.where(neg_a_mag < 0.1)  # was: 0.44
    # bump_vals = np.zeros_like(d.val)
    # bump_vals[bump_idx] = 0
    # bump_vals[bump_idx] = +0.1  # systematically increase / decrease low/high redshift moduli
    # d = d + ift.Field.from_raw(d.domain, arr=bump_vals)

    likelihood_energy = ift.GaussianEnergy(d, N.inverse) @ R
    ic_and_minimizers = (ic_sampling_lin, ic_sampling_nl, geoVI_sampling_minimizer, ic_newton, descent_finder,)

    mode = 'uniform_drawing' if uniform_drawing else 'exponential_drawing'
    LH_meta = _LhMetaContainer(d=d, neg_a_mag=neg_a_mag, s_model=s, s_mdl_meta=arguments,
                               x=x, ZP=X, init_pos=initial_pos, b0=init_fluctuations_parameter,
                               noise_cov=N, dataset_name="synthetic", mode=mode,
                               ic_and_minimizers=ic_and_minimizers, ground_truth_field=ground_truth_field)
    LH = _LhContainer(like=likelihood_energy, meta=LH_meta)
    return LH


