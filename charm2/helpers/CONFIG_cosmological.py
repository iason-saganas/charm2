import numpy as np
from dataclasses import dataclass
from .utilitites import *
import nifty8 as ift
from typing import Literal, Any

# Defining necessary spaces, fields and operators:
# Geometric object 'x' as signal space. The corresponding values (coordinate points) can be grabbed via x.field().val
# Set the resolution of the signal space
# Build the to-be-inferred signal from a correlated field + a line model
# Build the signal response as an operator chain '_OpChain'
# Read the distance moduli and redshifts
# Read the Covariance matrix from the data for the Gaussian Likelihood
# Define iteration and minimization control

# Read raw data. quantity_u := Union2.1 data; quantity_p := Pantheon+ data; quantity_d := DESY5 data.

__all__ = ["cosmological_likelihood", "ic_sampling_lin", "ic_sampling_nl", "geoVI_sampling_minimizer",
           "ic_newton", "descent_finder"]

from .custom_correlated_field import CustomSimpleCorrelatedField


def cosmological_likelihood(data_to_use:Literal["Union2.1", "Pantheon", "DESY5"], init_fluctuations_parameter):
    """

    :param data_to_use:                     String indicating which dataset to load.
    :param init_fluctuations_parameter:     Typically 0.05, 0.2 or 0.5. Not the associated latent variable but the
                                            parameter itself.
    :return:
    """
    z, mu, covariance = read_data(data_to_use)

    # print("Typical variance level: ", np.mean(np.diag(covariance)))
    # import matplotlib.pyplot as plt
    #
    # plt.errorbar(z, mu, yerr=np.diag(covariance), fmt='o', lw=0, elinewidth=1, capsize=3, markersize=2, color="black")
    # plt.xlabel("$z$")
    # plt.ylabel("$\mu(z)$")
    # plt.ylim(32, 46)
    # plt.show()

    config = {
        'Signal Field Resolution': 4096,  # 2**12
        'Length of signal space': np.max(np.log(1 + z)),
        'Factor to extend signal space size by': 2,
    }

    n_pix, x_length, x_fac = [float(setting) for setting in config.values()]
    n_dp = len(z)

    pxl_size = x_length / n_pix  # bin size

    x = ift.RGSpace(int(n_pix), distances=pxl_size)  # The signal space.
    x_ext = ift.RGSpace(int(x_fac * n_pix), distances=pxl_size)  # The signal space extended by the
    # in `config` specified factor.

    x = attach_custom_field_method(x)  # Attach the custom `field` method.
    x_ext = attach_custom_field_method(x_ext)  # Attach the custom `field` method.

    data_space = ift.UnstructuredDomain((n_dp,))

    neg_a_mag = np.log(1+z)  # The negative scale factor magnitude, x = -log(a) = log(1+z)

    # Arguments of the correlated field model
    args_cfm = {
        'offset_mean': 0,
        'offset_std': None,
        'fluctuations': (.2, .14),  # we use a uniform prior on the fluctuations, these arguments will be ignored
        'loglogavgslope': (-4, 1e-16),
        'asperity': None,
        'flexibility': None,
    }

    # Arguments of the line model
    args_lm = {
        #'slope': (2, 5),
        'slope': (10, 10),
        #'intercept': (30, 10)
        'intercept': (30, 30)
    }

    X = ift.FieldZeroPadder(domain=x, new_shape=(x_fac*n_pix, ))

    # The to-be-inferred signal on the extended domain
    s_cfm = CustomSimpleCorrelatedField(target=x_ext, **args_cfm, use_uniform_prior_on_fluctuations=True,
                                        tukey_taper_ends=True)
    s_line = LineModel(target=x_ext, args=args_lm)

    s = s_cfm + s_line

    s.line_model = s_line
    s.cf_model = s_cfm

    arguments = 'cfm_' + str(args_cfm) + '_lm_' + str(args_lm)

    # Build the signal response, noise operator, data field and others
    R = build_response(signal_space=x, signal=X.adjoint @ s, data_space=data_space, neg_scale_factor_mag=neg_a_mag)

    N = CovarianceMatrix(domain=data_space, matrix=covariance, sampling_dtype=np.float64, tol=1e-4)

    d = ift.Field(domain=ift.DomainTuple.make(data_space,), val=mu)

    # initial_pos = construct_initial_position(n_pix_ext=int(n_pix * x_fac), distances=pxl_size, fluctuations=0.2)
    initial_pos = construct_initial_position(n_pix_ext=int(n_pix * x_fac), distances=pxl_size, fluctuations=init_fluctuations_parameter)

    # FOR ANALYSIS OF POSSIBLE SYSTEMATIC EFFECTS. COMMENT OUT WHEN NO LONGER NEEDED:

    # bump_idx = np.where(neg_a_mag > 0.44)
    # bump_vals = np.zeros_like(d.val)
    # bump_vals[bump_idx] = +0.008

    # bump_vals[bump_idx] = -0.04
    # d = d + ift.Field.from_raw(d.domain, arr=bump_vals)

    likelihood_energy = ift.GaussianEnergy(d, N.inverse) @ R

    LH_meta = _LhMetaContainer(d=d, neg_a_mag=neg_a_mag,s_model=s, s_mdl_meta=arguments,
                               x=x, ZP=X, init_pos=initial_pos, b0=init_fluctuations_parameter,
                               noise_cov=covariance, dataset_name=data_to_use)
    LH = _LhContainer(like=likelihood_energy, meta=LH_meta)

    # In old versions, you might explicitly need these args:
    properties_old_versions = (LH.like, LH.meta.d, LH.meta.neg_a_mag, LH.meta.s_mdl_meta, LH.meta.x, LH.meta.ZP, LH.meta.s_model, LH.meta.init_pos, LH.meta.noise_cov, LH.meta.dataset_name)
    likelihood_energy, d, neg_a_mag, arguments, x, X, s, initial_pos, covariance, data_to_use = properties_old_versions
    return LH


# Iteration control for `MGVI` and linear parts of the inference
ic_sampling_lin = ift.AbsDeltaEnergyController(name="Precise linear sampling", deltaE=0.02, iteration_limit=100)

# Iteration control for `geoVI`
ic_sampling_nl = ift.AbsDeltaEnergyController(name="Coarser, nonlinear sampling", deltaE=0.5, iteration_limit=20,
                                              convergence_level=2)
# For the non-linear sampling part of geoVI, the iteration controller has to be "promoted" to a minimizer:
geoVI_sampling_minimizer = ift.NewtonCG(ic_sampling_nl)

# KL Minimizer control, the same energy criterion as the geoVI iteration control, but more iteration steps
ic_newton = ift.AbsDeltaEnergyController(name='Newton Descent Finder', deltaE=0.1, convergence_level=2,
                                         iteration_limit=35)
descent_finder = ift.NewtonCG(ic_newton)

raise_warning("\nUnion2.1 covariance matrix is only symmetric up to a factor of 10^{-10}.\n"
              "Pantheon+ covariance matrix is only symmetric up to a factor of 10^{-4}.\n\n")

