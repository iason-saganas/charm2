import matplotlib.pyplot as plt
import numpy as np
from data_storage.style_components.matplotlib_style import *
from utilitites import *
import nifty8 as ift

# Defining necessary spaces, fields and operators:
# Geometric object 'x' as signal space. The corresponding values (coordinate points) can be grabbed via x.field().val
# Set the resolution of the signal space
# Build the to-be-inferred signal from a correlated field
# Build the signal response as an operator chain '_OpChain'
# Read the distance moduli and redshifts
# Read the Covariance matrix from the data for the Gaussian Likelihood
# Define iteration and minimization control

config = {
    'Signal Field Resolution': 8000,
    'Length of signal space': 3,  # See comment in documentation of 'CustomRGSpace'.
    # 'Run Inference with Union2.1 data': True,
    # 'Run Inference with Pantheon+ data': False,
}

n_pix, x_length = [float(setting) for setting in config.values()]
z, mu, covariance = read_data_pantheon()
n_dp = len(z)

x = CustomRGSpace(n_pix, distances=x_length / n_pix)  # The Signal space.
data_space = ift.UnstructuredDomain((n_dp,))

neg_a_mag = np.log(1+z)  # The negative scale factor magnitude, x = -log(a) = log(1+z)

args = {
    'offset_mean': 0,
    'offset_std': None,
    'fluctuations': (1, 1e-16),
    'loglogavgslope': (-4, 1e-16),
    'asperity': None,
    'flexibility': None,
}

s = ift.SimpleCorrelatedField(target=x, **args)  # The to-be-inferred signal

# Build the signal response and noise operator
R = build_response(signal_space=x, signal=s, data_space=data_space, neg_scale_factor_mag=neg_a_mag)
N = CovarianceMatrix(domain=data_space, matrix=covariance, sampling_dtype=np.float64, tol=1e-4)


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

raise_warning("\nUnion2.1 covariance matrix is only symmetric up to a factor of 10^{-10}.\n"
              "Pantheon+ covariance matrix is only symmetric up to a factor of 10^{-4}.")
