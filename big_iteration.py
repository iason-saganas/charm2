import matplotlib.pyplot as plt
import numpy as np
from data_storage.style_components.matplotlib_style import *
from utilitites import *
import nifty8 as ift

config = {
        'Signal Field Resolution': 1024,  # 2**10 for the FFT
        'Length of signal space': 1,  # See comment in the documentation of 'CustomRGSpace'.
        'Factor to extend signal space size by': 3,
        # 'Run Inference with Union2.1 data': True,
        # 'Run Inference with Pantheon+ data': False,
    }

n_pix, x_length, x_fac = [int(setting) for setting in config.values()]
z, mu, covariance = read_data_union()
n_dp = len(z)
pxl_size = x_length / n_pix
x = ift.RGSpace(n_pix, distances=pxl_size)  # The signal space.
X = ift.FieldZeroPadder(domain=x, new_shape=(x_fac*n_pix, ))
pxl_size = x_length / n_pix
x = ift.RGSpace(n_pix, distances=pxl_size)  # The signal space.
x_ext = ift.RGSpace(x_fac * n_pix, distances=pxl_size)  # An extended signal space.
data_space = ift.UnstructuredDomain((n_dp,))
x = attach_custom_field_method(x)  # Attach the `field` method
x_ext = attach_custom_field_method(x_ext)  # Attach the `field` method

neg_a_mag = np.log(1+z)  # The negative scale factor magnitude, x = -log(a) = log(1+z)

# vary the offset mean from 15-50
# vary the offset sdt mean from 1-15
# vary Fix the offset sdt sdt to 1

def main(offset_mean, offset_sdt):


    args = {
        'offset_mean': offset_mean,
        'offset_std': (offset_sdt, 1),
        'fluctuations': (1, 0.1),
        'loglogavgslope': (-4, 0.1),
        'asperity': None,
        'flexibility': None,
    }

    s = ift.SimpleCorrelatedField(target=x_ext, **args)  # The to-be-inferred signal on the extended domain
    cf_parameters = str(args.values())

    # Build the signal response, noise operator, data field and others
    R = build_response(signal_space=x, signal=X.adjoint(s), data_space=data_space, neg_scale_factor_mag=neg_a_mag)
    N = CovarianceMatrix(domain=data_space, matrix=covariance, sampling_dtype=np.float64, tol=1e-4)
    d = ift.Field(domain=ift.DomainTuple.make(data_space,), val=mu)

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



    # ToDo: I need to understand the Pantheon+ Data better

    likelihood_energy = ift.GaussianEnergy(d, N.inverse) @ R
    global_iterations = 6

    posterior_samples = ift.optimize_kl(likelihood_energy=likelihood_energy,
                                        total_iterations=global_iterations,
                                        n_samples=kl_sampling_rate,
                                        kl_minimizer=descent_finder,
                                        sampling_iteration_controller=ic_sampling_lin,
                                        nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                                        output_directory='Output',
                                        return_final_position=True,
                                        resume=False)
    # Save the inference run
    pickle_me_this('big_iteration/Union2.1_data_cf_parameters_'+cf_parameters, posterior_samples)
    print('\nSaved posterior samples as ', 'Union2.1_data_cf_parameters_' + cf_parameters + '.pickle \n.'
          'Use "analyze_stored_data.py" to visualize results.\n')


counter = 0
for offset_mean in np.linspace(15, 50, 10):
    counter += 1
    print(counter)
    for offset_std in np.linspace(1, 15, 5):
        # main(offset_mean, offset_std)

        args = {
            'offset_mean': offset_mean,
            'offset_std': (offset_std, 1),
            'fluctuations': (1, 0.1),
            'loglogavgslope': (-4, 0.1),
            'asperity': None,
            'flexibility': None,
        }

        s = ift.SimpleCorrelatedField(target=x_ext, **args)  # The to-be-inferred signal on the extended domain

        import glob
        index = counter - 1
        directory = "data_storage/pickled_inferences/real/big_iteration"
        pickle = glob.glob(f"{directory}/*.pickle")[index]

        samples = unpickle_me_this(pickle)
        posterior_realizations_list, last_position_cf = samples
        s_mean, s_var = posterior_realizations_list.sample_stat(s)
        array_to_save = X.adjoint(s_mean).val
        with open(f'{directory}/savearr.txt', 'a') as f:
            f.write(np.array2string(array_to_save, separator=',', threshold=2000))
            f.write(',')  # Add a newline after each array

    if counter == 8:
        print("breaking")
        break


