import nifty8 as ift
from utilitites import *
import numpy as np

# Defining necessary spaces, fields and operators

config = {
    'Signal Field Resolution': 8000,
    'Number of synthetic datapoints': 200,
    'Length of signal space': 3,  # See comment in documentation of 'CustomRGSpace'.
    'Noise level': .05,
}

n_pix, n_dp, x_length, noise_level = [float(setting) for setting in config.values()]

x = CustomRGSpace(n_pix, distances=x_length / n_pix)  # The Signal space.
data_space = ift.UnstructuredDomain((n_dp,))

neg_a_mag = unidirectional_radial_los(n_dp)  # The negative scale factor magnitude, x = -log(a) = log(1+z)

args = {
    "offset_mean": 0,
    "offset_std": None,
    "fluctuations": (1.1, 1e-16),
    "loglogavgslope": (-4, 1e-16),
    "asperity": None,
    "flexibility": None
}
s = ift.SimpleCorrelatedField(target=x, **args)  # The to-be-inferred signal

# Build the signal response and noise operator

R = build_response(signal_space=x, signal=s, data_space=data_space, neg_scale_factor_mag=neg_a_mag)
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
