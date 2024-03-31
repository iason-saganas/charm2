import numpy as np
from utilitites import *
import nifty8 as ift

# Defining necessary spaces, fields and operators:
# Geometric object 'x' as signal space. The corresponding values (coordinate points) can be grabbed via x.field().val
# Set the resolution of the signal space
# Build the to-be-inferred signal from a correlated field + a line model
# Build the signal response as an operator chain '_OpChain'
# Read the distance moduli and redshifts
# Read the Covariance matrix from the data for the Gaussian Likelihood
# Define iteration and minimization control

# Read raw data. quantity_u := Union2.1 data; quantity_p := Pantheon+ data.
z_u, mu_u, covariance_u = read_data_union()
z_p, mu_p, covariance_p = read_data_pantheon()

config = {
    'Signal Field Resolution': 1024,  # 2**10
    'Length of signal space': np.max(np.log(1+z_p)),  # x_max. Has to be set manually according to which data is used
    'Factor to extend signal space size by': 1.3,
}

n_pix, x_length, x_fac = [float(setting) for setting in config.values()]
# n_dp_u = len(z_u)
n_dp_p = len(z_p)

pxl_size = x_length / n_pix  # bin size

x = ift.RGSpace(n_pix, distances=pxl_size)  # The signal space.
x_ext = ift.RGSpace(x_fac * n_pix, distances=pxl_size)  # The signal space extended by the in `config` specified factor.

x = attach_custom_field_method(x)  # Attach the custom `field` method.
x_ext = attach_custom_field_method(x_ext)  # Attach the custom `field` method.

# data_space_u = ift.UnstructuredDomain((n_dp_u,))
data_space_p = ift.UnstructuredDomain((n_dp_p,))


# neg_a_mag_u = np.log(1+z_u)  # The negative scale factor magnitude, x = -log(a) = log(1+z)
neg_a_mag_p = np.log(1+z_p)

# Arguments of the correlated field model
args_cfm = {
    'offset_mean': 0,
    'offset_std': None,
    'fluctuations': (1.8, 1.8),
    'loglogavgslope': (-4, 1e-16),
    'asperity': None,
    'flexibility': None,
}

# Arguments of the line model
args_lm = {
    'slope': (2, 5),
    'intercept': (30, 30)
}

X = ift.FieldZeroPadder(domain=x, new_shape=(x_fac*n_pix, ))

# The to-be-inferred signal on the extended domain
s = ift.SimpleCorrelatedField(target=x_ext, **args_cfm) + LineModel(target=x_ext, args=args_lm)
arguments = 'cfm_' + str(args_cfm) + '_lm_' + str(args_lm)

# Build the signal response, noise operator, data field and others
# R_u = build_response(signal_space=x, signal=X.adjoint @ s, data_space=data_space_u, neg_scale_factor_mag=neg_a_mag_u)
R_p = build_response(signal_space=x, signal=X.adjoint @ s, data_space=data_space_p, neg_scale_factor_mag=neg_a_mag_p)
# N_u = CovarianceMatrix(domain=data_space_u, matrix=covariance_u, sampling_dtype=np.float64, tol=1e-4)
N_p = CovarianceMatrix(domain=data_space_p, matrix=covariance_p, sampling_dtype=np.float64, tol=1e-4)
# d_u = ift.Field(domain=ift.DomainTuple.make(data_space_u,), val=mu_u)
d_p = ift.Field(domain=ift.DomainTuple.make(data_space_p,), val=mu_p)

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

