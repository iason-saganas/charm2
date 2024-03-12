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
    'Signal Field Resolution': 1024,  # 2**10 for the FFT
    'Length of signal space': 1,
    'Factor to extend signal space size by': 6,
    # 'Run Inference with Union2.1 data': True,
    # 'Run Inference with Pantheon+ data': False,
}

n_pix, x_length, x_fac = [int(setting) for setting in config.values()]
z, mu, covariance = read_data_union()
n_dp = len(z)

pxl_size = x_length / n_pix
x = ift.RGSpace(n_pix, distances=pxl_size)  # The signal space.
x_ext = ift.RGSpace(x_fac * n_pix, distances=x_fac*pxl_size)  # An extended signal space.
data_space = ift.UnstructuredDomain((n_dp,))
x = attach_custom_field_method(x)  # Attach the `field` method
x_ext = attach_custom_field_method(x_ext)  # Attach the `field` method

neg_a_mag = np.log(1+z)  # The negative scale factor magnitude, x = -log(a) = log(1+z)

args = {
    'offset_mean': 30,
    'offset_std': None,
    'fluctuations': (0.6, 0.1),
    'loglogavgslope': (-4, 0.5),
    'asperity': None,
    'flexibility': None,
}

X = ift.FieldZeroPadder(domain=x, new_shape=(x_fac*n_pix, ))

alpha = ift.NormalTransform(mean=2, sigma=3, key="alpha")
beta = ift.NormalTransform(mean=0, sigma=3, key="beta")

contraction = ift.ContractionOperator(domain=x, spaces=None)
alpha = contraction.adjoint @ alpha
beta = contraction.adjoint @ beta
x_op = ift.DiagonalOperator(diagonal=x.field())
line_model = x_op @ alpha + beta
line_model_ext = X @ line_model

s = ift.SimpleCorrelatedField(target=x_ext, **args) + line_model_ext # The to-be-inferred signal on the extended domain
cf_parameters = str(args.values())

# ift.plot_priorsamples(line_model)

# Build the signal response, noise operator, data field and others
R = build_response(signal_space=x, signal=X.adjoint(s), data_space=data_space, neg_scale_factor_mag=neg_a_mag)
N = CovarianceMatrix(domain=data_space, matrix=covariance, sampling_dtype=np.float64, tol=1e-4)
d = ift.Field(domain=ift.DomainTuple.make(data_space,), val=mu)

plot_op = ift.DomainChangerAndReshaper(domain=R.target, target=ift.DomainTuple.make(ift.RGSpace(R.target.shape)))

# ift.plot_priorsamples(s)
# ift.plot_priorsamples(plot_op @  R)

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

# ToDo: Please delete these following lines in the future.

