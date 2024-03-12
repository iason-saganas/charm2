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

config = {
    'Signal Field Resolution': 2048,
    'Number of synthetic datapoints': 200,
    'Length of signal space': 1,
    'Fac to extend signal space by': 3,
    'Noise level': .05,
}

n_pix, n_dp, x_length, x_fac, noise_level = [float(setting) for setting in config.values()]

x = ift.RGSpace(n_pix, distances=x_length / n_pix)  # The Signal space.
x = attach_custom_field_method(x)  # Attach `field()` method
x_ext = ift.RGSpace(x_fac*n_pix, distances=x_length / n_pix)
x_ext = attach_custom_field_method(x_ext)  # Attach `field()` method
data_space = ift.UnstructuredDomain((n_dp,))

neg_a_mag = unidirectional_radial_los(n_dp)  # The negative scale factor magnitude, x = -log(a) = log(1+z)

args = {
    "offset_mean": 30,
    "offset_std": None,
    "fluctuations": (1.1, 1e-16),
    "loglogavgslope": (-4, 1),
    "asperity": None,
    "flexibility": None
}
s = ift.SimpleCorrelatedField(target=x_ext, **args)  # The to-be-inferred signal

# Build the signal response and noise operator
X = ift.FieldZeroPadder(domain=x, new_shape=(int(x_fac*n_pix), ))
R = build_response(signal_space=x, signal=X.adjoint(s), data_space=data_space, neg_scale_factor_mag=neg_a_mag)
N = ift.ScalingOperator(data_space, noise_level, np.float64)

# Iteration control for `MGVI` and linear parts of the inference
ic_sampling_lin = ift.AbsDeltaEnergyController(name="Precise linear sampling", deltaE=0.05, iteration_limit=100)

# Iteration control for `geoVI`
ic_sampling_nl = ift.AbsDeltaEnergyController(name="Coarser, nonlinear sampling", deltaE=0.5, iteration_limit=20,
                                              convergence_level=2)
# For the non-linear sampling part of geoVI, the iteration controller has to be "promoted" to a minimizer:
geoVI_sampling_minimizer = ift.NewtonCG(ic_sampling_nl)

# KL Minimizer control, the same energy criterion as the geoVI iteration control, but more iteration steps
ic_newton = ift.AbsDeltaEnergyController(name='Newton Descent Finder', deltaE=0.5, convergence_level=2,
                                         iteration_limit=35)
descent_finder = ift.NewtonCG(ic_newton)
