# import nifty8 as ift
from typing import Literal

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

from dataclasses import dataclass

__all__ = ["synthetic_likelihood"]


def synthetic_likelihood(init_fluctuations_parameter, data_generation_args:DataArgs, ground_truth_args:GroundTruthArgs,
                         mode:Literal["non-parametric", "flat_LCDM", "flat_EDE"]):
    """

    :param init_fluctuations_parameter: np.float,   The initial fluctuations parameter
                                                    the the real DESY5 dataset.
    :param data_generation_args: DataArgs,          The data generation arguments, see also documentation of class `DataArgs`
    :param ground_truth_args: GroundTruthArgs,      The ground truth generation arguments, see also documentation of class `GroundTruthArgs`
    :param mode                                     Determines whether to fit with a flat LCDM or evolving DE or non-parametric
                                                    model.
    :return:
    """
    dga = data_generation_args
    n_dp, use_des_like_data_distribution, uniform_drawing = (dga.n_dp, dga.use_des_like_data_distribution,
                                                             dga.uniform_drawing)
    if use_des_like_data_distribution:
        des_hist_edges = np.loadtxt(f"{data_dir}/desy5_data_histogram_bin_edges.txt")
        des_hist_counts = np.loadtxt(f"{data_dir}/desy5_data_histogram_bin_counts.txt")
        n_dp = 1829
        neg_a_mag = unidirectional_radial_los(n_dp, uniform_drawing=True, end_of_data=0.7514160887, specific_hist=(des_hist_edges, des_hist_counts))  # The negative scale factor magnitude,
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


    X = ift.FieldZeroPadder(domain=x, new_shape=(x_fac*n_pix, ))

    # Signal model
    if mode == "non-parametric":

        # Construct initial position
        if init_fluctuations_parameter is None:
            raise ValueError("Please provide init_fluctuations_parameter if in non-parametric mode")

        initial_pos = construct_initial_position(n_pix_ext=int(n_pix * x_fac), distances=pxl_size,
                                                 fluctuations=init_fluctuations_parameter,
                                                 apply_prior_xi_s=True)  # `apply_prior_xi_s=True` <-> draws random init pos
        args_cfm = {
            'offset_mean': 0,
            'offset_std': None,
            'loglogavgslope': (-4, 1e-16),
            'fluctuations': (.2, .14),  # Will be overriden by `use_uniform_prior_on_fluctuations` argument
            'asperity': None,
            'flexibility': None,
        }
        args_lm = {
            'slope': (2, 5),
            'intercept': (30, 10)
        }

        # Construct model
        s_cfm = CustomSimpleCorrelatedField(target=x_ext, **args_cfm, use_uniform_prior_on_fluctuations=True,
                                            tukey_taper_ends=False)
        s_line = LineModel(target=x_ext, args=args_lm)
        s = s_cfm + s_line
        s.line_model = s_line
        s.cf_model = s_cfm

        arguments = 'cfm_' + str(args_cfm) + '_lm_' + str(args_lm)

    elif mode == "flat_LCDM":
        s = FlatLCDM(target=x_ext)

        scalar_domain = ift.DomainTuple.make(())
        init_pos_dict = {"flat_lcdm_H0": ift.makeField(scalar_domain, arr=np.random.standard_normal()),
                         "flat_lcdm_Omega_m": ift.makeField(scalar_domain, arr=np.random.standard_normal()), }

        initial_pos = ift.MultiField.from_dict(dct=init_pos_dict)

        arguments = 'H0_[0,100]_uniform_and_Omega_m_[0,1]_uniform'
    elif mode == "flat_EDE":
        s = FlatEDE(target=x_ext)

        scalar_domain = ift.DomainTuple.make(())
        init_pos_dict = {"EDE_Omega_m": ift.makeField(scalar_domain, arr=np.random.standard_normal()),
                         "EDE_Omega_H0": ift.makeField(scalar_domain, arr=np.random.standard_normal()),
                         "EDE_w0": ift.makeField(scalar_domain, arr=np.random.standard_normal()),
                         "EDE_wa": ift.makeField(scalar_domain, arr=np.random.standard_normal()),
                         }

        initial_pos = ift.MultiField.from_dict(dct=init_pos_dict)
        arguments = 'w0_[-2,2]_uniform_and_wa_[-15,15]_uniform_and_Omega_m_[0,1]_uniform_and_H0_[0,100]_uniform'
    else:
        raise ValueError(f"Unknown mode {mode}")


    # The ground truth model
    if ground_truth_args.mode == "flat_LCDM":
        Ωm0 = ground_truth_args.Ωm0
        offset = signal_from_H0(Ωm0)
        s_g = PiecewiseLinear(signal_space=x_ext, omega_m_custom=Ωm0, omega_l_custom=1-Ωm0, high_curv=False,
                              offset_custom=offset)
    elif ground_truth_args.mode == "flat_EDE":
        custom_values = {
            'H0': ground_truth_args.H0,
            'w0': ground_truth_args.w0,
            'wa': ground_truth_args.wa,
            'Ωm': ground_truth_args.Ωm0
        }
        s_g = FlatEDE(target=x_ext, custom_parameter_values=custom_values)
    elif ground_truth_args.mode == "non-parametric":
        args_gr_cfm = {
            'offset_mean': 0,
            'offset_std': None,
            'loglogavgslope': (-4, 1e-16),
            'fluctuations': (ground_truth_args.b0, 1e-16),  # will NOT be overriden
            'asperity': None,
            'flexibility': None,
        }
        offset = signal_from_H0(ground_truth_args.H0)
        args_gr_lm = {
            'slope': (ground_truth_args.m0, 1e-16),
            'intercept': (offset, 1e-16)
        }
        s_g_cfm = CustomSimpleCorrelatedField(target=x_ext, **args_gr_cfm, use_uniform_prior_on_fluctuations=False,
                                            tukey_taper_ends=False)
        s_g_line = LineModel(target=x_ext, args=args_gr_lm)
        s_g = s_g_cfm + s_g_line
    else:
        raise ValueError("Unknown mode")

    # Build the signal response, noise operator, data field and others
    R = build_response(signal_space=x, signal=X.adjoint @ s, data_space=data_space, neg_scale_factor_mag=neg_a_mag)
    R_g = build_response(signal_space=x, signal=X.adjoint @ s_g, data_space=data_space, neg_scale_factor_mag=neg_a_mag)
    # Ground truth response
    N = ift.ScalingOperator(domain=(data_space, ), factor=noise_level, sampling_dtype=np.float64)


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

    if data_generation_args.use_des_like_data_distribution:
        mode_info = 'des_like_drawing_'
    elif uniform_drawing:
        mode_info = 'uniform_drawing_'
    else:
        mode_info = 'exponential_drawing_'
    mode_info += 'ground_is_' + ground_truth_args.mode  # 'flat_EDE', 'flat_LCDM' or 'non-parametric'
    mode_info += '_while_s_model_is_' + mode
    LH_meta = _LhMetaContainer(d=d, neg_a_mag=neg_a_mag, s_model=s, s_mdl_meta=arguments,
                               x=x, ZP=X, init_pos=initial_pos, b0=init_fluctuations_parameter,
                               noise_cov=N, dataset_name="synthetic", mode=mode_info,
                               ic_and_minimizers=ic_and_minimizers, ground_truth_field=ground_truth_field,
                               ground_truth_args=ground_truth_args, data_generation_args=dga,)
    LH = _LhContainer(like=likelihood_energy, meta=LH_meta)
    return LH


