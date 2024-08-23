import nifty8 as ift
from utilitites import *
import numpy as np

# Defining necessary spaces, fields and operators:
# Geometric object 'x' as signal space. The corresponding values (coordinate points) can be grabbed via x.field().val
# Set the resolution of the signal space
# Build the to-be-inferred signal from a correlated field
# Build the signal response as an operator chain '_OpChain'
# Construct synthetic lines of sight
# Define the Covariance of the Gaussian Likelihood as a 'ScalingOperator'
# Define iteration and minimization control

n_dp = 500
neg_a_mag = unidirectional_radial_los(n_dp, uniform_drawing=True)  # The negative scale factor magnitude,
# x = -log(a) = log(1+z)

config = {
    'Signal Field Resolution': 4096,
    'Length of signal space': np.max(neg_a_mag),
    'Fac to extend signal space by': 1.3,
    'Noise level': .1,
}

n_pix, x_length, x_fac, noise_level = [float(setting) for setting in config.values()]

pxl_size = x_length / n_pix
x = ift.RGSpace(n_pix, distances=pxl_size)  # The Signal space.
x_ext = ift.RGSpace(x_fac*n_pix, distances=pxl_size)  # The extended signal space
x = attach_custom_field_method(x)  # Attach `field()` method
x_ext = attach_custom_field_method(x_ext)  # Attach `field()` method
data_space = ift.UnstructuredDomain((n_dp,))

# Arguments of the correlated field model
args_cfm = {
    'offset_mean': 0,
    'offset_std': None,
    'fluctuations': (.4, .2),
    'loglogavgslope': (-4, 1e-16),
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
s = ift.SimpleCorrelatedField(target=x_ext, **args_cfm) + LineModel(target=x_ext, args=args_lm)
ift.plot_priorsamples(s)
# The ground truth model
s_g = PiecewiseLinear(signal_space=x_ext, omega_m_custom=0.3, omega_l_custom=2)

arguments = 'cfm_' + str(args_cfm) + '_lm_' + str(args_lm)

# Build the signal response, noise operator, data field and others
R = build_response(signal_space=x, signal=X.adjoint @ s, data_space=data_space, neg_scale_factor_mag=neg_a_mag)
R_g = build_response(signal_space=x, signal=X.adjoint @ s_g, data_space=data_space, neg_scale_factor_mag=neg_a_mag)
# Ground truth response
N = ift.ScalingOperator(domain=(data_space, ), factor=noise_level, sampling_dtype=np.float64)

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
