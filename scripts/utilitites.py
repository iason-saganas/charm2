import matplotlib.pyplot as plt
from nifty8.domains.rg_space import RGSpace
from nifty8.domain_tuple import DomainTuple
from nifty8.field import Field
import numpy as np
from scipy.constants import c, G
from scipy.optimize import curve_fit
from typing import Tuple
from nifty8.operators.diagonal_operator import DiagonalOperator
from nifty8.operators.adder import Adder
from nifty8.library.los_response import LOSResponse
from nifty8.operators.operator import _OpChain
from nifty8.domains.unstructured_domain import UnstructuredDomain
from nifty8.operators.contraction_operator import ContractionOperator
from nifty8.operators.normal_operators import NormalTransform, LognormalTransform, StandardUniformTransform
from data_storage.style_components.matplotlib_style import *
import pandas as pd
from nifty8.operators.matrix_product_operator import MatrixProductOperator
import warnings
import pickle
from nifty8.sugar import plot_priorsamples, from_random
from nifty8.utilities import lognormal_moments
import gzip
import os
import shutil
import datetime
import matplotlib.gridspec as gridspec
from scipy.stats import norm

__all__ = ['create_plot_1', 'unidirectional_radial_los', 'build_response', 'kl_sampling_rate', 'read_data_union',
           'read_data_pantheon', 'CovarianceMatrix', 'raise_warning', 'build_flat_lcdm', 'pickle_me_this',
           'unpickle_me_this', 'current_expansion_rate', 'attach_custom_field_method', 'chi_square_dof',
           'build_charm1_agnostic', 'plot_comparison_fields', 'show_plot', 'plot_flat_lcdm_fields',
           'plot_charm1_in_comparison_fields', 'LineModel', 'plot_synthetic_ground_truth', 'plot_synthetic_data',
           'PiecewiseLinear', 'plot_charm2_in_comparison_fields', 'plot_prior_distribution', 'calculate_approximate_mode',
           'read_data_des', 'store_meta_data', 'get_datetime', 'read_data', 'plot_lognormal_histogram',
           'plot_prior_cfm_samples', 'posterior_parameters', 'visualize_posterior_histograms',
           'construct_initial_position', 'evolving_dark_energy_fit']


def LineModel(target: RGSpace, args: dict, custom_slope: float = None):
    """
    The line is also defined on the extended domain.
    :param custom_slope:      If specified, line will have fixed custom slope (useful for construction of ground truth).
    :param args:              The dictionary that contains the mean and standard deviations of the parameters of the
                              line model (slope and offset).
    :param target:            The extended domain for which to build the model.
                              Custom field method needs to be attached.
    """
    x = target
    contraction = ContractionOperator(domain=x, spaces=None)
    alpha_mean, alpha_std = args['slope']
    beta_mean, beta_std = args['intercept']
    if custom_slope is not None:
        alpha = NormalTransform(mean=custom_slope, sigma=1e-16, key="line model slope")
    else:
        alpha = NormalTransform(mean=alpha_mean, sigma=alpha_std, key="line model slope")
    beta = NormalTransform(mean=beta_mean, sigma=beta_std, key="line model y-intercept")
    alpha = contraction.adjoint @ alpha  # distribute the one alpha value that is drawn over the whole domain
    beta = contraction.adjoint @ beta
    x_coord = DiagonalOperator(diagonal=x.field())
    line_model = x_coord @ alpha + beta
    line_model.myAttr = "Line test"
    return line_model


def attach_custom_field_method(space: RGSpace):
    """
    Takes an instance of `RGSpace`, a regular cartesian grid, and attaches a new method called `field` which can then
    be called on the `RGSpace` instance.
    The `field` method returns a field of the values associated with the `distances` property of the raw,
    geometric object `RGSpace`.
    Example:

    Let the domain of a signal field be a regular, 1D grid space of length 100 and consisting of 5000 pixels.
    Then, each pixel has a width of 0.02.

    Calling this method will return a `NIFTy` field object whose values are [0., 0.02, 0.04, ..., 99.98].
    This way, the distance values of the domain may be, e.g., added to the signal field, as needed in the signal
    response of `CHARM`.
    In `NIFTy`'s, `RGSpace`, the field values are defined at the beginning of the voxels, i.e., the length in the above
    example, 100, is only reached in the limit of infinitesimal distances (this can be seen clearly through plotting
    prior samples with low resolution).

    :param space: RGSpace,      The regular grid space to attach the 'field' method to.
    :return: space:             The regular grid with the method attached.
    """

    def field(rg_space):
        dimension = len(rg_space.shape)
        if dimension != 1:
            raise ValueError(f"Class method 'field' only works for 1D grid spaces, \n"
                             f"not for {dimension}D grid spaces.")

        values = []
        resolution = rg_space.shape[0]
        distances = rg_space.distances[0]
        for iindex in range(0, resolution):
            values.append(iindex * distances)
        dom = DomainTuple.make((rg_space,))
        return Field(dom, val=np.array(values))

    space.field = lambda: field(space)
    return space


def unidirectional_radial_los(n_los: int, uniform_drawing=False) -> np.ndarray:
    """
    Generates an ordered array of normally distributed random numbers. Returns that array ('ends') and the 'starts'
    array (zeroes). Represents synthetic redshifts.
    :param uniform_drawing:     If true, draws from distributes the data uniformly over redshift space and doesnt
                                normalize the data.
                                If false, draws more realistically from a lognormal distribution.
    :param n_los:   int,        Number of lines-of-sight to construct.
    :return: ends: np.array,    The synthetic lines-of-sight.
    """
    n_los = int(n_los)
    if uniform_drawing:
        arr = 1.2*np.random.rand(n_los)
        # arr = 2*np.random.rand(n_los)
        ends = np.sort(arr)
    else:
        end_of_data = 1.2
        # arr = 1.2*np.random.lognormal(mean=0, sigma=0.9, size=n_los)
        arr = end_of_data*np.random.exponential(scale=2, size=n_los)
        # arr = 1.2*np.random.lognormal(mean=1, sigma=0.2, size=n_los) + np.append(0.1*np.random.rand(10),(np.zeros(n_los-10)))
        maximum = np.max(arr)
        ends = end_of_data*np.sort(arr / maximum)
    return ends


def build_response(signal_space: RGSpace, signal: _OpChain, data_space: UnstructuredDomain,
                   neg_scale_factor_mag: np.array) -> _OpChain:
    """
    Builds the `CHARM` signal response as a chain of operators.
    :param signal_space:            'RGSpace',              The grid the signal is defined over.
    :param signal:                  '_OpChain',             The (correlated field + line) modelling the signal whose
                                                            target is 'signal_space'.
    :param data_space:              'UnstructuredDomain',   The data space.
    :param neg_scale_factor_mag:    np.array,               The measured negative scale factor mags.
    :return: R: _OpChain, The signal response.
    """
    s = signal
    x = signal_space
    neg_a_mag = neg_scale_factor_mag
    K = (8 * np.pi * G / 3) ** (-1 / 2) * c * 1000
    n_dp = len(neg_a_mag)

    los_integration = LOSResponse(domain=x, starts=[np.zeros(int(n_dp))], ends=[neg_a_mag])
    offset_by_x_tilde = Adder(a=x.field(), neg=False)
    e_to_the_power_of_x = DiagonalOperator(diagonal=Field.from_raw(data_space, np.exp(neg_a_mag)))
    offset_by_negative_five = Adder(a=5, neg=True, domain=DomainTuple.make(data_space, ))

    R_s = offset_by_negative_five(5 * np.log10(e_to_the_power_of_x(K * los_integration(
        np.exp(offset_by_x_tilde(-1 / 2 * s))))))
    return R_s


def create_plot_1(x, y, color, x_label, y_label, title, x_lim=None, y_lim=None, ls=".", legend_label=None):
    """
    Creates a standard plot.
    """
    if x is None:
        # Drawing log scale power spectrum against log k modes
        x = np.log(range(len(y) + 1)[1:len(y) + 1])  # Cut out first element, which is zero for some reason and shift 1
        # to the right
    if legend_label is not None:
        plt.plot(x, y, ls, color=color, label=legend_label)
        plt.legend()
    else:
        plt.plot(x, y, ls, color=color)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_all_synthetic_post_kl(x_field_val, s_mean_val, sqrt_s_var_val, ground_truth_signal_val, neg_a_mag):
    plt.errorbar(x_field_val, s_mean_val, yerr=sqrt_s_var_val, fmt='-', color=blue, ecolor=light_blue,
                 label=r'Reconstruction (lightblue: $1\sigma$')
    plt.plot(x_field_val, ground_truth_signal_val, '-', color='black', label='Ground Truth')
    plt.xlabel(r'Negative Scale Factor Magnitude $x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$')
    plt.ylabel('Signal Field $s(x)$')
    plt.title('Reconstruction Vs. Ground Truth')
    # plt.xlim(0, 1)
    # plt.ylim(-0.8, 3)
    plt.plot(neg_a_mag, 2 * np.ones(len(neg_a_mag)), ".", lw=0, color="black", label='Distribution Of Datapoints')
    plt.legend()
    plt.show()

    plt.errorbar(x_field_val, s_mean_val - ground_truth_signal_val, yerr=sqrt_s_var_val, fmt='-', color=blue,
                 ecolor=light_blue, label=r'$1\sigma$')
    plt.errorbar(x_field_val, s_mean_val - ground_truth_signal_val, color=blue, yerr=2 * sqrt_s_var_val, fmt='-',
                 ecolor=lighter_blue, label=r'$2\sigma$')
    plt.errorbar(x_field_val, s_mean_val - ground_truth_signal_val, color=blue, yerr=3 * sqrt_s_var_val, fmt='-',
                 ecolor=lightest_blue, label=r'$3\sigma$')
    plt.xlabel(r'Negative Scale Factor Magnitude $x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$')
    plt.ylabel(r'Residuals $\Delta s(x)$')
    plt.title('Reconstructed Signal Field Deviation From Ground Truth')
    plt.plot(neg_a_mag, 0.75 * np.ones(len(neg_a_mag)), ".", lw=0, color=light_red, label='Distribution Of Datapoints')
    plt.legend()
    # plt.xlim(0, 1)
    # plt.ylim(-0.5, 1)
    plt.hlines(0, 0, 1, linestyles='--', color="black")
    plt.show()


def kl_sampling_rate(index: int):
    """
    Callable for sampling of KL. KL minimization can be performed on its samples instead of computing it directly.
    First, get a ballpark, in later iteration increase sampling rate.
    :param index:
    :return:
    """
    if index < 2:
        return 7
    else:
        return 15


def read_data_union():
    """
    Source: `https://supernova.lbl.gov/Union/figures/SCPUnion2.1_mu_vs_z.txt`
    The data is not ordered but will be after this function is called.
    Note how on reordering of the data, the covariance matrix must be reordered as well.
    :return:
    """
    print("\nReading Union2.1 data...")
    path_to_data = 'data_storage/raw_data/Union_2_1_data.txt'
    path_to_cov = 'data_storage/raw_data/Union2_1_Cov-syst.txt'
    z = np.loadtxt(path_to_data, usecols=[1], dtype=float)  # read data
    sorting_indices = np.argsort(z)  # get sorting indices (from smallest to biggest), s.t. mu can be sorted in that
    # order as well
    z = z[sorting_indices]  # sort from smallest to biggest
    mu = np.loadtxt(path_to_data, usecols=[2], dtype=float)[sorting_indices]
    covariance = np.loadtxt(path_to_cov, dtype=float)
    covariance = covariance[sorting_indices][:, sorting_indices]  # Reorder!
    print("\tFinished")
    return z, mu, covariance


def read_data_pantheon():
    """
    Source: https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR
    `zHD` is the Hubble Diagram redshift, including CMB and peculiar velocity corrections.
    The data is already ordered in ascending order.
    :return:
    """
    print("\nReading Pantheon+ data...")
    path_to_data = 'data_storage/raw_data/Pantheon+SH0ES.csv'
    path_to_cov = 'data_storage/raw_data/Pantheon+SH0ES_STAT+SYS.txt'

    required_fields = ["zHD", "MU_SH0ES"]
    df = pd.read_csv(path_to_data, sep=" ", skipinitialspace=True, usecols=required_fields)
    z = np.array(df.zHD)
    mu = np.array(df.MU_SH0ES)

    df = pd.read_table(path_to_cov)
    arr = np.array(df["1701"])  # the first line corresponds to '1701' = length of the data array
    covariance = arr.reshape(1701, 1701)
    print("\tFinished")
    return z, mu, covariance


def read_data_des():
    """
    Copied from DES-SN5YR file `SN_only_cosmosis_likelihood.py` from GitHub.
    See comment at the end of the page of `https://github.com/des-science/DES-SN5YR/tree/main/4_DISTANCES_COVMAT`.
    """
    print("\nReading DESY5 data...")
    filename = "data_storage/raw_data/DES-SN5YR_HD.csv"
    data = pd.read_csv(filename, delimiter=",", comment='#')
    # The only columns that we actually need here are the redshift,
    # distance modulus and distance modulus error

    ww = (data['zHD'] > 0.00)
    # use the vpec corrected redshift for zCMB
    zCMB = data['zHD'][ww]
    # distance modulus and relative stat uncertainties
    mu_obs = data['MU'][ww]
    mu_obs_err = data['MUERR_FINAL'][ww]

    filename = "data_storage/raw_data/DES-SN5YR-STAT+SYS.txt.gz"
    # This data file is just the systematic component of the covariance -
    # we also need to add in the statistical error on the magnitudes
    # that we loaded earlier
    with gzip.open(filename, "rt") as f:
        line = f.readline()
        n = int(line)
        cov_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov_mat[i, j] = float(f.readline())

    # Now add in the statistical error to the diagonal
    for i in range(n):
        cov_mat[i, i] += mu_obs_err[i] ** 2
    f.close()

    cov_mat = cov_mat[ww][:, ww]

    # Finally, reorder:
    zCMB = zCMB.values
    mu_obs = mu_obs.values

    ordering_indices = np.argsort(zCMB)
    zCMB = zCMB[ordering_indices]
    mu_obs = mu_obs[ordering_indices]
    cov_mat = cov_mat[ordering_indices][:, ordering_indices]
    print("\tFinished")

    return zCMB, mu_obs, cov_mat


def read_data(data_to_use):
    if data_to_use == "Union2.1":
        return read_data_union()
    elif data_to_use == "Pantheon+":
        return read_data_pantheon()
    elif data_to_use == "DESY5":
        return read_data_des()
    else:
        raise ValueError(f"Can't read unknown data <{data_to_use}>")


def spectral_decompose(matrix: np.ndarray, tol: float) -> Tuple:
    """
    Let 'matrix' be A, a strictly positive, symmetric matrix. Then, it has a decomposition via the spectral
    theorem:

        A = U D U^†

    Here, D = diag(λ_1, λ_2,...λ_n) the diagonal matrix containing the spectral values and Λ the unitary
    transformation matrix containing the eigenvectors.
    This function checks for positivity and symmetry and returns the triple U, D, U^†.

    @param matrix: np.ndarray, The matrix to be decomposed.
    @param tol: float, The absolute tolerance to sanity check the symmetry and recomposition of the decomposed matrix.

    :return: U D U^†, tuple, The decomposition
    """
    # Note that here, we use 'np.linalg.eigh' which is a routine explictly for symmetric matrices.
    A = np.array(matrix)
    B = A.T

    # print("Decimals after the comma, A: ", str(A[0, 1]).split(".")[1])
    # print("Decimals after the comma, A.T: ", str(B[0, 1]).split(".")[1])

    if not np.allclose(A, B, atol=tol, rtol=0):
        raise ValueError("Won't spectral decompose non symmetric matrix.")
    rows, columns = A.shape
    if rows != columns:
        raise ValueError("Matrix to spectral decompose is not square.")
    lambdas, U = np.linalg.eigh(A)
    if np.any(lambdas < 0):
        raise ValueError("Matrix is not positive definite.")
    D = np.diag(lambdas)
    U_inv = np.linalg.inv(U)  # Since U is unitary, U^† = U^{-1}
    sanity_check = np.allclose(A, U @ D @ U_inv, atol=tol, rtol=0)
    if not sanity_check:
        raise ValueError("Something went wrong during the diagonalization process.")
    return U, D, U_inv


class CovarianceMatrix(MatrixProductOperator):
    """
    The same as 'MatrixProductOperator' but has an additional argument 'sampling_dtype' which is needed to sample from
    this operator if it is used in a likelihood and also provides two additional functionalities:

    N = CustomMatrixProductOperator(some_domain, some_matrix)
    N.inverse provides the inverted matrix

    Furthermore, 'tol' needs to be provided, which is the absolute tolerance used for sanity checking the symmetry
    of the matrix and its eigen-decomposition and other operations.

    Finally, 'enable_transformation' specifies whether the method 'get_sqrt()', needed for the `geoVI` transformation,
    can be accessed or not.

    E.g. 'enable_transformation=True' is unnecessary when first instantiating the nominal covariance matrix N.
    'enable_transformation=True' is mandatory when constructing the likelihood and calling N.inverse.

    """

    def __init__(self, domain, matrix, tol, spaces=None, flatten=False, sampling_dtype=None,
                 enable_transformation=False):
        super().__init__(domain, matrix, spaces=spaces, flatten=flatten)
        self._dtype = sampling_dtype
        self.tol = tol
        self.enable_transformation = enable_transformation
        if enable_transformation:
            self.mtrx_sqrt = self.get_fct()

    @property
    def inverse(self):
        # Return N^{-1}, such that N^{-1} @ N ~ 1. The off-diagonal elements are of the order 10e-18 -> 10e-20.
        return CovarianceMatrix(domain=self.domain, matrix=np.linalg.inv(self._mat), tol=self.tol,
                                enable_transformation=True, sampling_dtype=self.sampling_dtype)

    def get_fct(self):
        """
        Let self._mat = A be a strictly positive, symmetric matrix. Then, it has a unique matrix square root via
        the spectral theorem:

            √A^{-1} = U √D U^†,

        where √D:=diag(√λ_1, √λ_2, ..., √λ_n). This method calls the function 'spectral_decompose' (which checks for
        positivity and symmetry) and returns matrix square root √A.

        After having called N.inverse in the likelihood, a new Covariance Matrix is returned that represents the
        inverse N^{-1} of the original covariance matrix N. In that case, `NIFTy` calls the get_sqrt() method of
        the N^{-1} operator, i.e. it returns  √N^{-1}.

        """
        if not self.enable_transformation:
            raise ValueError("Can't get factor of Covariance matrix, `enable_transformation` is set to False.")
        A = self._mat
        U, D, U_inv = spectral_decompose(A, tol=self.tol)
        square_root = U @ np.sqrt(D) @ U_inv
        sanity_check = np.allclose(A, square_root @ square_root, atol=self.tol, rtol=0)
        if not sanity_check:
            raise ValueError("A is not approximately equal to the square of its"
                             "square root decomposition.")
        return square_root

    def get_sqrt(self):
        if not self.enable_transformation:
            raise ValueError("Can't get square root of Covariance matrix, `enable_transformation` is set to False.")
        """
        Returns the matrix square root √N^{-1} as new 'MatrixProductOperator'.
        """
        return MatrixProductOperator(self._domain, self.mtrx_sqrt)

    def draw_sample(self, from_inverse=False):
        if from_inverse:
            raise NotImplementedError
        n_dp = self._mat.shape[0]
        noise_vector = np.random.multivariate_normal(mean=np.zeros(n_dp), cov=self._mat, check_valid='warn')
        d_space = UnstructuredDomain(n_dp)
        return Field(domain=DomainTuple.make(d_space), val=noise_vector)


def raise_warning(message):
    warnings.warn(message, category=Warning)


def build_charm1_agnostic(mode="union2.1"):
    """
    Let s':=log(ρ/ρ0) be the signal field defined in charm1, https://arxiv.org/abs/1608.04007.
    Let s:=log(ρ) be the signal field defined in charm2.
    The transformation between them is

    s = log(ρ0 exp(s') ),

    where ρ0 = 3H0^2/8πG and Porqueres et al. assumed a value of H0 = 68.6 km/s/Mpc.
    :param mode: str,       One of 'union2.1' or 'pantheon+'.
    """
    if mode == "union2.1":
        path_to_x = "data_storage/raw_data/x_field_charm1_union2_1.txt"
        path_to_s = "data_storage/raw_data/s_field_charm1_union2_1.txt"
        path_to_D = r"data_storage/raw_data/D_field_charm1_union2_1.txt"
        correction_offset = 0.0246747  # log(rho/rho0) in charm1 is not precisely at 0, which must be incorrect
    elif mode == "pantheon+":
        path_to_x = "data_storage/raw_data/x_field_charm1_pantheon+.txt"
        path_to_s = "data_storage/raw_data/s_field_charm1_pantheon+.txt"
        path_to_D = r"data_storage/raw_data/D_field_charm1_pantheon+.txt"
        correction_offset = 0.10272  # log(rho/rho0) in charm1 is not precisely at 0, which must be incorrect
    elif mode == "pantheon+_reformulated":
        path_to_x = "data_storage/raw_data/charm1_reformulated/x_field_charm1_pantheon+.txt"
        path_to_s = "data_storage/raw_data/charm1_reformulated/s_field_charm1_pantheon+.txt"
        path_to_D = r"data_storage/raw_data/charm1_reformulated/D_field_charm1_pantheon+.txt"
    elif mode == "des_reformulated":
        path_to_x = "data_storage/raw_data/charm1_reformulated/x_field_charm1_des.txt"
        path_to_s = "data_storage/raw_data/charm1_reformulated/s_field_charm1_des.txt"
        path_to_D = r"data_storage/raw_data/charm1_reformulated/D_field_charm1_des.txt"
    else:
        raise ValueError("Unrecognized mode in `build_charm1_agnostic`.")

    if mode == "union2.1" or mode == "pantheon+":
        x = np.loadtxt(path_to_x)
        s_prime = np.loadtxt(path_to_s) - correction_offset
        D = np.loadtxt(path_to_D)

        rho0 = 3 * 68.6 ** 2 / (8 * np.pi * G)
        s = np.log(rho0 * np.exp(s_prime))

        s_err = np.sqrt(D)
    else:
        x = np.loadtxt(path_to_x)
        s = np.loadtxt(path_to_s)
        D = np.loadtxt(path_to_D)
        s_err = np.sqrt(D)

    return x, s, s_err


def build_flat_evolving_dark_energy(x: np.array, H_0=68.37, w_a=-8.8, w_0=-0.36, omega_m0=0.495):

    # Values are from this paper: https://arxiv.org/pdf/2401.02929
    # H_0 = 68.37 # Self-chosen!
    # H_0 = 73.7 # Self-chosen!
    # w_a = -8.8
    # w_0 = -0.36
    # omega_m0 = 0.495

    # Another paper: https://arxiv.org/pdf/2503.06712
    # H_0 = 67.8
    # w_a = -1.37
    # w_0 = -0.67
    # omega_m0 = 0.31

    omega_l0 = 1 - omega_m0
    m = 3 / (8 * np.pi * G)
    E_sq = omega_m0*np.exp(3*x) + omega_l0 * np.exp(3*x*(1+w_0+w_a)) * np.exp( -3*(w_a*(1-np.exp(-x))))
    inner_log_func = m * H_0 ** 2 * E_sq
    s_base = np.log(inner_log_func)

    return s_base


def evolving_dark_energy_fit(x, s):
    popt = curve_fit(build_flat_evolving_dark_energy, x, s)
    print("popt: ", popt[0])


def build_flat_lcdm(x: np.array, mode: str):
    """
    References:
        - https://arxiv.org/pdf/1807.06209.pdf, equations (14) and (15) and table (2) of the Planck 2018 paper
        Contains constraints with TT, TE, EE + lowE + lensing, standing for: CMB temperature power spectrum, high
        multipole spectrum, polarization spectrum + its likelihood + lensing of the CMB photons by the gravitational
        gradients
        - https://arxiv.org/pdf/2202.04077.pdf, table (3) of the Pantheon+ Analysis Cosmology paper.
        Values for FlatΛCDM from SN alone.

    It should be noted that when comparing these datasets:

    "These estimates are highly model dependent and this needs to be borne in mind when comparing with other
    measurements, for example the direct measurements of H0 discussed in Sect 5.4" from the planck 2018 paper.
    We use these fields to get a feeling for the tendency of direct measurements to lie higher than CMB inferred values.

    A flat universe is assumed, k=0 and the radiation density today is well approximated by 0.
    This function returns the field values as well as the 1σ uncertainty of this base signal field:

    s_base(x) = log( m * hat{H_0}^2 * (Ω_Λ + Ω_m e^(3x) ) ),

    where m is a constant, m := 3/(8π) * (G/[G])^(-1) and hat{H_0} is H_0/(km/s/Mpc).
    Since in a flat universe a constraint in Ω_m is translatable into a constraint for Ω_Λ, above expression is
    equivalent to

    s_base(x) = log( m * hat{H_0}^2 * (1 + Ω_m(e^(3x)-1) ) ).

    The uncertainty is a function of
    x itself and is calculated via the Gaussian Propagated Error:

    Δs_base(x) = (Σ_i (( ∂ s_base(x) / ∂ x ) Δx ) ^2 )^(1/2),

    where the sum runs over all uncertain values, i.e. Ω_m and H_0.
    Attention!
    This error propagation should only be used to get a sense of the scale of the 1σ error contour of these fields.
    It assumes that there is no degeneracy between H0 and Ω_m (in the sense that they are independent variables) and
    that s_base is well approximated by its linearization.

    The values used are:
    ------------------
    Full CMB Analysis
    H_0 = 67.36 ±  0.54
    Ω_m = 0.3153 ± 0.0073
    Ω_Λ = (1 - Ω_m) ± 0.0073 = 0.6847 ± 0.0073

    ------------------
    Direct measurement using SN + including Cepheid host distances and covariance
    H_0 = 73.60 ±  1.1
    Ω_m = 0.334 ± 0.018
    Ω_Λ = (1 - Ω_m) ± 0.0073 = 0.666 ± 0.018

    :param x: np.array,     The x-coordinate to construct the field over
    :param mode: str,       One of 'CMB' or 'SN', determines which cosmological values to use.

    """
    if mode == 'SN':
        H_0 = 73.6
        delta_H_0 = 1.1

        omega_m = 0.334
        delta_omega_m = 0.018
    elif mode == 'CMB':
        H_0 = 67.36
        delta_H_0 = 0.54

        omega_m = 0.3153
        delta_omega_m = 0.0073
    else:
        raise ValueError('Unknown mode in `build_flat_lcdm`.')

    # Construct the base lcdm, cmb parameters signal field
    m = 3 / (8 * np.pi * G)
    inner_log_func = m * H_0 ** 2 * (1 + omega_m * (np.exp(3 * x) - 1))
    s_base = np.log(inner_log_func)

    # Construct its 1σ uncertainty array
    common_denominator = 1 / inner_log_func
    partial_omega_m = common_denominator * m * H_0 ** 2 * (np.exp(3 * x) - 1)
    partial_H_0 = common_denominator * 2 * m * H_0 * (1 + omega_m * (np.exp(3 * x) - 1))

    # Order: First Ω_m, then H0
    all_errors = [delta_omega_m, delta_H_0]
    all_derivatives = [partial_omega_m, partial_H_0]

    gaussian_error_to_sum = np.zeros(len(x))
    for der, er in zip(all_derivatives, all_errors):
        gaussian_error_to_sum += (der * er) ** 2

    gaussian_error = np.sqrt(gaussian_error_to_sum)

    return s_base, gaussian_error


def pickle_me_this(filename: str, data_to_pickle: object):
    path = "data_storage/pickled_inferences/" + filename + ".pickle"
    file = open(path, 'wb')
    pickle.dump(data_to_pickle, file)
    file.close()


def unpickle_me_this(filename: str, absolute_path=False):
    if absolute_path:
        file = open(filename, 'rb')
    else:
        file = open("data_storage/pickled_inferences/" + filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def store_meta_data(name, duration_of_inference, len_d, inference_type, signal_model_param,
                    global_kl_iterations, expansion_rate="", data_storage_dir_name="temp"):
    """
    Stores metadata related to an inference run in a text file and manages associated data files.

    This function creates a metadata file named using the current date and time. The file contains
    details about the inference run, including the type of data used, the length of the dataset, and
    the duration of the inference in minutes. The function also appends the content of additional files
    located in a temporary directory to the metadata file, then moves the metadata file to a specified
    directory and deletes the temporary directory.

    Args:
        name (str): How the metadata file should be called, e.g. a datetime string
        duration_of_inference (float): The duration of the inference in seconds.
        len_d (int): The length of the dataset used for the inference.
        inference_type (str): A string indicating the type of inference ('synthetic' or 'real').
        signal_model_param (str): A string containing the inference parameter values of the used cfm + line model.
        global_kl_iterations (int): The number of global kl minimization runs.
        expansion_rate: (str): Information about the H0 estimate from the reconstruction (optional)
        data_storage_dir_name: Where the intermediate output information from `optimize_kl` is stored.
        If temp (=> Synthetic inference) the folder is deleted.
        If cache (=> Real inference), the folder is not deleted to ensure proper functionality of the `resume` arg of
        `optimize_kl`.

    """
    # Create file name with datetime
    filename = f"metadata_{name}.txt"

    # Convert duration from seconds to minutes
    duration_minutes = duration_of_inference / 60

    # Define paths
    temp_dir = f'data_storage/pickled_inferences/{data_storage_dir_name}'
    final_dir = f'data_storage/pickled_inferences/{inference_type}'

    # Ensure final directory exists
    os.makedirs(final_dir, exist_ok=True)

    # Create the metadata file
    with open(filename, 'w') as file:
        file.write(f"Charm2 inference run on the {transform_datetime_string(name)}. Mode: {inference_type}\n")
        file.write(f"Length of dataset: {len_d}\n")
        file.write(f"Time took in minutes: {duration_minutes:.2f}\n")
        file.write(f"Model parameters: {signal_model_param}\n")
        file.write(f"{expansion_rate}\n")
        file.write(f"Number of global KL minimization runs: {global_kl_iterations}\n\n")

        # Append contents of the other files
        for additional_file in ['counting_report.txt', 'minisanity.txt']:
            additional_file_path = os.path.join(temp_dir, additional_file)
            if os.path.exists(additional_file_path):
                with open(additional_file_path, 'r') as afile:
                    file.write(afile.read())
                file.write("\n\n")  # Add line breaks between sections
            else:
                file.write(f"File {additional_file} not found.\n\n")

    # Move the metadata file to the final directory
    shutil.move(filename, os.path.join(final_dir, filename))

    # Delete the temp directory and its contents
    if data_storage_dir_name == "temp":
        shutil.rmtree(temp_dir)
    else:
        # Don't remove files called f`cache{dataset_used}` for `resume` functionality of `optimize_kl` to work
        pass


def get_datetime():
    """
    Returns the current datetime as a string with microseconds stripped off,
    spaces replaced by '_', and colons replaced by '-'.

    Returns:
        str: Formatted datetime string.
    """
    # Get the current datetime without microseconds
    now = datetime.datetime.now().replace(microsecond=0)

    # Convert to string and replace spaces and colons
    formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")

    return formatted_datetime


def transform_datetime_string(datetime_str):
    """
    Transform the datetime string by replacing underscores with spaces and hyphens in the time part with colons.

    Args:
        datetime_str (str): The datetime string to transform.

    Returns:
        str: The transformed datetime string.
    """
    # Replace underscores with spaces
    datetime_str = datetime_str.replace('_', ' ')

    # Replace hyphens with colons in the time part
    date_part, time_part = datetime_str.split(' ', 1)
    time_part = time_part.replace('-', ':')

    # Combine the date part and the transformed time part
    return f"{date_part} {time_part}"


def current_expansion_rate(s: np.array, delta_s: np.array = None) -> float:
    """
    The relationship between the expansion rate and the signal is:

    H(s)^2 / (km/s/Mpc)^2 = 8πG/[G]/3 * e^s(x),

    in other words the Hubble constant H0 = H(s0) = H(s(x=0)) is

    H0 / (km/s/Mpc)  = sqrt( 8πG/[G]/3 * e^s0).

    The returned value is H0 in km/s/Mpc rounded to two decimal places.
    If the standard deviation is specified, the Gaussian error of H0 is returned as well.
    The Gaussian error is calculated as

    ΔH0 = ∂H0(s0)/∂s0 • Δs0

    :param s: np.array,         The inferred signal according to the definition s:=log(ρ).
    :param delta_s: np.array,   The inferred signal uncertainty (standard deviation).

    """
    s0 = s[0]
    H0 = np.sqrt(8 * np.pi * G / 3 * np.exp(s0))
    if delta_s is None:
        return np.round(H0, 2)
    else:
        delta_s0 = delta_s[0]
        delta_H = 1 / 2 * (8 * np.pi * G / 3 * np.exp(s0)) ** (1 / 2) * delta_s0
        return np.round(H0, 2), np.round(delta_H, 5)


def chi_square_dof(real, model, inv_cov):
    # Compute squared differences, input the inverse noise covariance to calculate the chi^2_dof
    # Degrees of freedom are the number of datapoints - 2 for loglogavgslope and fluctuations
    # Althoug really, loglogavgslope is fixed in charm2.
    raise_warning("Probably improper normalization of the chi^2 by the DOF.")
    res = (real - model)
    return (res.T @ inv_cov @ res) / (len(real) - 2)


def build_comparison_fields():
    """
    Constructs comparison fields for plots.
    """
    x = np.linspace(0, 2, 1000)
    x_sparse = np.linspace(0, 2, 40)
    s_cmb, s_cmb_err = build_flat_lcdm(x, mode='CMB')
    s_cmb_sparse, _ = build_flat_lcdm(x_sparse, mode='CMB')
    s_sn, s_sn_err = build_flat_lcdm(x, mode='SN')
    s_sn_sparse, _ = build_flat_lcdm(x_sparse, mode='SN')
    H0_sn = current_expansion_rate(s=s_sn)
    H0_cmb = current_expansion_rate(s=s_cmb)

    x_coordinates = [x, x_sparse]
    cmb = [s_cmb, s_cmb_err, s_cmb_sparse, H0_cmb]
    sn = [s_sn, s_sn_err, s_sn_sparse, H0_sn]
    return x_coordinates, cmb, sn


def plot_comparison_fields(plot_fluctuations_scale_visualization=False, ax_object=None):
    """
    Adds comparison fields to a plot, but does not show it.

    :param plot_fluctuations_scale_visualization: bool, if true, adds and shows a plot of linear fits to the comparison
    signal fields, as well as the residuals between said fits and the actual curves.
    This represents the point-wise fluctuations around the offset the correlated field needs to model.
    We take thus take the fluctuation parameter to be the square root of the mean squared residuals.
    :param ax_object=None: A mpl ax object, if passed, instead of plt.plot, ax_object.plot will be called

    :return: handles: tuple,    A tuple containing three strings, representing x label, y label and title for the plot
                                that can be fed into the function `show_plot()`.
    """
    x_coordinates, cmb, sn = build_comparison_fields()
    x, x_sparse = x_coordinates
    s_cmb, s_cmb_err, s_cmb_sparse, H0_cmb = cmb
    s_sn, s_sn_err, s_sn_sparse, H0_sn = sn
    # dash_dot_dotted = (0, (3, 5, 1, 5, 1, 5))
    dash_dot_dotted = (0, (15, 5))
    # long_dash_with_offset = (5, (10, 3))
    long_dash_with_offset = (0, (10, 5))

    orange = (0.902, 0.624, 0.0)
    light_orange = (0.902, 0.624, 0.0, 0.2)

    green = (0.0, 0.62, 0.45)
    light_green = (0.0, 0.62, 0.45, 0.2)

    if ax_object:
        ax_object.errorbar(x=x, y=s_cmb, yerr=s_cmb_err, fmt="None",
                           ecolor=light_green)
        ax_object.errorbar(x=x, y=s_sn, yerr=s_sn_err, fmt="None",
                           ecolor=light_orange)
        ax_object.plot(x_sparse, s_cmb_sparse, ls=dash_dot_dotted, lw="2", color="green",
                 label=r'$s_{\mathrm{CMB}}$. $\hat{H}_0=' + str(H0_cmb) + '$', markersize=0
                 )
        ax_object.plot(x_sparse, s_sn_sparse, ls=long_dash_with_offset, lw="2", color="orange",
                 label=r'$s_{\mathrm{SN}}$. $\hat{H}_0=' + str(H0_sn) + '$', markersize=0
                 )
    else:
        plt.errorbar(x=x, y=s_cmb, yerr=s_cmb_err, fmt="None",
                     ecolor=light_orange)
        plt.errorbar(x=x, y=s_sn, yerr=s_sn_err, fmt="None",
                     ecolor=light_green)
        plt.plot(x_sparse, s_cmb_sparse, ls=dash_dot_dotted, lw="1", color=green,
                 label=r'$s_{\mathrm{CMB}}$. $\hat{H}_0=' + str(H0_cmb) + '$',
                 )
        plt.plot(x_sparse, s_sn_sparse, ls=long_dash_with_offset, lw="1", color=orange,
                 label=r'$s_{\mathrm{SN}}$. $\hat{H}_0=' + str(H0_sn) + '$',
                 )
    t = r"Flat $\Lambda$CDM Signal Fields." + "\nComparison Between CMB And Supernovae Measurements."
    xl = r"$x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$"
    yl = r"$s(x)$"
    handles = (xl, yl, t)
    if plot_fluctuations_scale_visualization:

        def linear(arg, m, y0):
            return m * arg + y0

        popt_sn = curve_fit(linear, x, s_sn)[0]
        popt_cmb = curve_fit(linear, x, s_cmb)[0]

        linear_fit_line_s_cmb = linear(x, *popt_cmb)
        linear_fit_line_s_sn = linear(x, *popt_sn)

        square_residuals_s_cmb = (s_cmb - linear_fit_line_s_cmb) ** 2
        square_residuals_s_sn = (s_sn-linear_fit_line_s_sn)**2

        variance_s_cmb = np.mean(square_residuals_s_cmb)
        variance_s_sn = np.mean(square_residuals_s_sn)

        plt.plot(x, linear_fit_line_s_cmb, "b-", label="Linear curve fit through $s_{cmb}$ field")
        plt.plot(x, linear_fit_line_s_sn, "r-", label="Linear curve fit through $s_{sn}$ field")
        plt.plot(x, square_residuals_s_cmb, "b.",label=r"Square residuals between $s_{cmb}$ and its best fit line")
        plt.plot(x, square_residuals_s_sn, "r.", label=r"Square residuals between $s_{sn}$ and its best fit line")
        plt.hlines(variance_s_cmb, 0, 2, color="b", ls="--", label="Mean squared residual (variance) $s_{cmb}$")
        plt.hlines(variance_s_sn, 0, 2, color="r", ls="--", label="Mean squared residual (variance) $s_{sn}$")
        print("Real signal std deviation s_cmb vs s_sn: ", np.sqrt(variance_s_cmb), np.sqrt(variance_s_sn))
        plt.legend()
        plt.show()
    return handles


def show_plot(x_lim: tuple = None,
              y_lim: tuple = None,
              save_filename: str = "",
              show: bool = True,
              title: str = "",
              x_label: str = "",
              y_label: str = "",
              loc: str="upper right",
              disable_legend = False):
    """
    Shows the currently constructed plot.
    :param x_lim: tuple,        The limits on the x-axis.
                                x_max should be at the maximum registered data point, e.g., the max pantheon+ value
                                for np.log(1+z).
    :param y_lim: tuple,        The limits on the y-axis.
    :param save_filename: str,  If not `""`, plot is saved under `save_filename`.
    :param show: bool,          If `False`, `plt.show()` is suppressed and the figure cleared.
    :param title: str,          The title of the plot.
    :param x_label: str,        The x label of the plot.
    :param y_label: str,        The y label of the plot.
    :return:
    """
    if not disable_legend:
        plt.legend(loc=loc)
    if x_lim is not None:
        plt.xlim(*x_lim)
    if y_lim is not None:
        plt.ylim(*y_lim)
    if title != "":
        plt.title(title)
    if x_label != "":
        plt.xlabel(x_label, fontsize=30)
    if y_label != "":
        plt.ylabel(y_label, fontsize=30)
    if save_filename != "":
        plt.tight_layout(pad=2)
        plt.savefig(save_filename + ".png", pad_inches=1)
    if show:
        plt.tight_layout(pad=2)
        plt.show()
    else:
        plt.clf()


def plot_flat_lcdm_fields(x_max: float, show: bool = False, save: bool = True):
    """
    Saves a plot of the flat ΛCDM model comparison fields applied for Planck 2018 CMB values and the Pantheon+ SN
    compilation values.
    :param save:
    :param x_max:       The upper limit to show on the x axis.
                        Should correspond to the logarithm of the (highest measured redshift + 1).
    :param show:        If False, plot is not shown.
    :return:
    """
    xl, yl, t = plot_comparison_fields()
    if save:
        filename = "data_storage/figures/comparison_flat_lcdm_cmb_sn_fields"
    else:
        filename = ""
    show_plot(x_lim=(0, x_max), y_lim=(29.5, 32.5), x_label=xl, y_label=yl, title="",
              save_filename=filename, show=show)


def plot_charm2_in_comparison_fields(x: np.array, s: np.array, s_err: np.array, x_max_pn: float, x_max_union: float,
                                     x_max_des: float, dataset_used: str, neg_a_mag, show: bool = False,
                                     save: bool = True, additional_samples = None):
    """
    Plots the reconstructed charm1 curve using Union2.1 data into a figure containing CMB and SN comparison fields.
    :param dataset_used:    Either `Union2.1`, `Pantheon+` or `DESY5`
    :param s_err:           The charm2 posterior standard deviation values.
    :param s:               The charm2 posterior mean values.
    :param x:               The coordinate axis of the charm2 reconstruction.
    :param x_max_union:     A vertical line is plotted at this point, indicating the end of the dataset.
    :param x_max_pn:        The max scale factor magnitude of the pantheon analysis.
    :param save:            Bool, whether or not to save the plot
    :param show:            Bool, whether or not to show the plot
    :param neg_a_mag:       The `x` array used in order to plot a histogram of it in the upper panel
    :param additional_samples: Additional fields to add to the plot (e.g. all posterior samples).
                               Needs to be a list of arrays.
    :return:
    """

    # Create a figure
    fig = plt.figure(figsize=(10, 8))

    # Create a GridSpec with 3 rows and 1 column
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Create the first subplot, occupying the first row
    ax1 = fig.add_subplot(gs[0])

    # Create the second subplot, occupying the second and third rows
    ax2 = fig.add_subplot(gs[1:])

    # Histogram of data
    ax1.hist(neg_a_mag, histtype="step", color="black", lw=0.5, bins=10)
    ax1.set_ylabel(r"$\#$ of dtps")
    ax1.set_xlim(0, x_max_pn)

    # Reconstruction in signal space
    xl, yl, t = plot_comparison_fields(ax_object=ax2)

    if dataset_used == "Union2.1":
        # ax2.vlines(x_max_union, 0, 50, linestyles='dashed', label="End of Union2.1 data")
        ax2.vlines(x_max_union, 0, 50, linestyles='dashed', label="End of data")
    elif dataset_used == "DESY5":
        # ax2.vlines(x_max_des, 0, 50, linestyles='dashed', label="End of DESY5 data")
        ax2.vlines(x_max_des, 0, 50, linestyles='dashed', label="End of data")
    else:
        pass
    current_expansion_mean, current_expansion_err = current_expansion_rate(s, s_err)
    h0_charm2 = str(current_expansion_mean)
    now = get_datetime()
    if save:
        filename = f"data_storage/figures/charm2_reconstruction_{dataset_used}_{now}"
    else:
        filename = ""

    # For visualization (transparency) purposes the field values are cut in half two times
    x_reduced = remove_every_second(x, n=2)
    s_reduced = remove_every_second(s, n=2)
    s_err_reduced = remove_every_second(s_err, n=2)
    # ax2.errorbar(x=x_reduced, y=s_reduced, yerr=s_err_reduced, color=blue, ecolor=light_blue,
    #              label=r"\texttt{charm2}, " + dataset_used + r" data. $\hat{H}_0=" + h0_charm2 + "$", markersize=1)
    ax2.errorbar(x=x_reduced, y=s_reduced, yerr=s_err_reduced, color=blue, ecolor=light_blue,
                 label=r"\texttt{charm2}, " + dataset_used, markersize=1)

    if additional_samples is not None:
        for s_sample in additional_samples:
            s_red = remove_every_second(s_sample, n=2)
            ax2.plot(x_reduced, s_red, color="orange", alpha=1, markersize=0, lw=2)


    # Evolving dark energy analysis, comment out when done
    # s_evolving = build_flat_evolving_dark_energy(x_reduced)
    # plt.plot(x_reduced, s_evolving, color="black", ls="-", alpha=1, lw=2, markersize=0, label="Evolving dark energy")

    special_legend_III()

    # For DES: ylim=(29.5, 32.5)
    plt.tight_layout()
    show_plot(x_lim=(0, x_max_pn), y_lim=(29.5, 32.5), x_label=xl, y_label=yl, title="",
              save_filename=filename, show=show, loc="upper left", disable_legend=True)


def plot_charm1_in_comparison_fields(show: bool = False, save: bool = True):
    """
    Plots the reconstructed charm1 curve using Union2.1 or Pantheon+ data into a figure containing CMB and SN
    comparison fields.
    :param save:
    :param x_max_union:     A vertical line is plotted at this point, indicating the end of the dataset.
    :param x_max_pn:        The max scale factor magnitude of the pantheon analysis.
    :param show:
    :return:
    """

    z_u, mu_u, _ = read_data_union()
    z_p, mu_p, _ = read_data_pantheon()

    neg_a_mag_u = np.log(1 + z_u)
    neg_a_mag_p = np.log(1 + z_p)

    x_max_pn = np.max(neg_a_mag_p)
    x_max_union = np.max(neg_a_mag_u)

    # Create a figure
    fig = plt.figure(figsize=(10, 8))

    # Create a GridSpec with 3 rows and 1 column
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Create the first subplot, occupying the first row
    ax1 = fig.add_subplot(gs[0])

    # Create the second subplot, occupying the second and third rows
    ax2 = fig.add_subplot(gs[1:])

    # Histogram of data
    ax1.hist(neg_a_mag_u, histtype="step", color="black", lw=0.5, bins=10)
    raise_warning("Probably wrong histogram shown")
    ax1.set_ylabel(r"$\#$ of dtps")
    ax1.set_xlim(0, x_max_pn)

    xl, yl, t = plot_comparison_fields(ax_object=ax2)
    # ax2.vlines(x_max_union, 0, 50, linestyles='dashed', label="End of Union2.1 data")
    ax2.vlines(x_max_union, 0, 50, linestyles='dashed', label="End of data")
    x, s, s_err = build_charm1_agnostic(mode="union2.1")
    h0_charm1 = str(current_expansion_rate(s))
    if save:
        filename = "data_storage/figures/PAPER_charm1_Union21_reformulated.png"
    else:
        filename = ""
    x_reduced = remove_every_second(x, 1)
    s_reduced = remove_every_second(s, 1)
    s_err_reduced = remove_every_second(s_err, 1)
    # ax2.errorbar(x=x_reduced, y=s_reduced, yerr=s_err_reduced, color=blue, ecolor=light_blue,
    #              label=r"\texttt{charm1}, Union2.1 data. $\hat{H}_0=" + h0_charm1 + "$",
    #              markersize=1)
    ax2.errorbar(x=x_reduced, y=s_reduced, yerr=s_err_reduced, color=blue, ecolor=light_blue,
                 label=r"\texttt{charm1}, Union2.1",
                 markersize=1)
    special_legend_III()
    show_plot(x_lim=(0, x_max_pn), y_lim=(29.5, 32.5), x_label=xl, y_label=yl, title="",
              save_filename=filename, show=show, loc="upper left", disable_legend=True)


def plot_synthetic_ground_truth(x: RGSpace, ground_truth: np.ndarray, x_max_pn: float, show=True, save=True,
                                reconstruction: tuple = None):
    """
    Plots the synthetic ground truth in signal space.
    :param reconstruction:      A tuple containing (reconstruction_mean, reconstruction_std)
    :param show: bool,          Whether to show the plot
    :param save: bool,          Whether to save the plot
    :param x_max_pn:            The max scale factor magnitude of the pantheon analysis.
    :param ground_truth:        The randomly drawn ground truth
    :param x: `RGSpace`
    :return:
    """
    x = x.field().val
    plt.vlines(1.2, 25, 40.5, linestyles='dashed', label="End of data", color="black",)
    xl = r"$x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$"
    yl = r"$s(x)$"
    if save:
        filename = "data_storage/figures/synthetic_ground_truth_standard_LCDM"
    else:
        filename = ""
    plt.plot(x, ground_truth, "-", color="black", lw=1, label="Synthetic ground truth")
    if reconstruction is not None:
        # For visualization purposes, the arrays are cut in half two times
        # such that the errorbars look transparent.
        signal_domain_reduced = remove_every_second(x, n=2)
        signal_field_reduced = remove_every_second(reconstruction[0], n=2)
        error_field_reduced = remove_every_second(reconstruction[1], n=2)
        plt.errorbar(x=signal_domain_reduced, y=signal_field_reduced, yerr=error_field_reduced, color=blue, ecolor=light_blue,
                     label=r"$\texttt{charm2}$ reconstruction", markersize=1)

    # Ensure the legend uses the same dash pattern
    special_legend_I()

    # Revert to: (0, 1.25) and ylim (32, 36.5) for figure limits in paper
    show_plot(x_lim=(0, 1.25), y_lim=(31.25, 35), x_label=xl, y_label=yl, title="",
              save_filename=filename, show=show, loc="upper left", disable_legend=True)


def special_legend_I():
    # Modifies some existing plot elements for better control (e.g. vertical lines without handles at the ends)
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Modify only the dashed line's handle
    handles = [
        plt.Line2D([0], [0], linestyle="--", color="black", markersize=0) if label == "End of data" else h
        for h, label in zip(handles, labels)
    ]

    # Recreate the legend with the modified handles
    plt.legend(handles, labels, loc="upper left", fontsize=25)


def remove_every_second(arr, n):
    """
    Removes every second element from the array, repeated n times.

    Parameters:
    arr (np.ndarray): The input array.
    n (int): The number of times to remove every second element.

    Returns:
    np.ndarray: The resulting array after n iterations.
    """
    for _ in range(n):
        arr = np.delete(arr, np.arange(1, arr.size, 2))
    return arr


def plot_synthetic_data(neg_scale_fac_mag: np.ndarray, data: np.ndarray, x_max_pn: float, mu_array: np.array,
                        show=True, save=True):
    """
    Plots the synthetic ground truth and data it creates.
    :param show: bool,              Whether to show the plot
    :param save: bool,              Whether to save the plot
    :param mu_array: np.array,     The real distance moduli used to get mu_min and mu_max for plotting purposes.
    :param x_max_pn: float,     The max scale factor magnitude of the pantheon analysis.
    :param neg_scale_fac_mag:   The unidirectional line of sights x=np.log(1+z)
    :param data:                The constructed synthetic data
    :return:
    """
    mu_min = np.min(mu_array)
    mu_max = np.max(mu_array)

    x_min = np.min(neg_scale_fac_mag)
    x_max = np.max(neg_scale_fac_mag)

    xl = r"$x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$"
    yl = r"$\mu (x)$"
    if save:
        filename = "data_storage/figures/synthetic_data_standard_LCDM"
    else:
        filename = ""

    plt.subplot(2, 1, 1)
    plt.hist(neg_scale_fac_mag, histtype="step", color="black", lw=0.5, bins=10)
    plt.ylabel(r"$\#$ of datapoints")
    plt.xlim(-0.1+x_min, x_max+0.1)
    plt.subplot(2, 1, 2)
    plt.plot(neg_scale_fac_mag, data, ".", color="black", label="Synthetic data points")
    show_plot(x_lim=(-0.1+x_min, x_max+0.1), y_lim=(mu_min-0.1, mu_max+0.1), x_label=xl, y_label=yl, title="",
              save_filename=filename, show=show, loc="upper left")


def PiecewiseLinear(signal_space: RGSpace, omega_m_custom: float = None, omega_l_custom: float = None,
                    high_curv=True):
    """
    A generative line model for a piecewise linear function with two contributions.
    Represents a LCDM model with an offset of around 30.
    :param omega_l_custom:  Contribution through cosmological constant
    :param omega_m_custom:  Contribution through matter.
    :param signal_space:    The signal regular grid space
    :return:
    """
    x = signal_space
    expander = ContractionOperator(domain=x, spaces=None).adjoint
    if high_curv:
        x_coord = DiagonalOperator(diagonal=Field(domain=DomainTuple.make(x), val=np.exp(5 * x.field().val))) # component goes with ~a^-5 (fictitious for high curvature)
    else:
        x_coord = DiagonalOperator(diagonal=Field(domain=DomainTuple.make(x), val=np.exp(3 * x.field().val)))  # matter goes with ~a^-3

    omega_m, m_deviation = (1, 1)
    omega_l, l_deviation = (1, 1)
    if omega_m_custom is not None and omega_l_custom is not None:
        omega_m, m_deviation = (omega_m_custom, 1e-16)
        omega_l, l_deviation = (omega_l_custom, 1e-16)
    contribution1 = expander @ LognormalTransform(omega_m, m_deviation, key="omega m contribution", N_copies=0)
    contribution2 = expander @ LognormalTransform(omega_l, l_deviation, key="omega l contribution", N_copies=0)
    # adder = Adder(a=Field(domain=DomainTuple.make(x), val=30 * np.ones(len(x.field().val))))
    offset = expander @ NormalTransform(30, 5, key="offset of piecewise linear")
    return np.log(x_coord(contribution1) + contribution2) + offset


def plot_prior_distribution(mean_std_tuple, n_samples=50, distribution_name="normal"):
    """
    Plot distribution of some hyperparameter x that has mean and standard deviation μ
    and σ according to 'mean_std_tuple'.
    :param mean_std_tuple:      μ, σ = mean_std_tuple
    :param n_samples:           The number of samples to plot.
                                Default: 50
    :param distribution_name:   Either 'normal' or 'lognormal'
    :return:
    """
    mean, sigma = mean_std_tuple
    log_mean, log_sigma = lognormal_moments(mean, sigma)
    key = "distribution operator for plot"
    if distribution_name == "normal":
        op = NormalTransform(mean, sigma, key=key)
    elif distribution_name == "lognormal":
        op = NormalTransform(log_mean, log_sigma, key=key).ptw("exp")
    else:
        raise ValueError("Unknown distribution")
    plot_priorsamples(op, n_samples=n_samples)


def plot_lognormal_histogram(mean: float, sigma: float, n_samples: int, vlines: np.array = None, save=False, show=True,
                             color = "black", mode="Lognormal"):
    """
    Plots a histogram visualizing the moment-matched lognormal transform.
    If `vlines` is provided, vertical lines will be drawn at the specified x-locations.
    Usage:

    plot_lognormal_histogram(mean=.06, sigma=0.03, n_samples=10000, vlines=[0.023, 0.05], save=True, show=True)

    :param mean:        The mean from which logmean is calculated with logsigma's help.
    :param sigma:       The sigma from which logsigma is calculated.
    :param n_samples:   How many samples to plot
    :param vlines:      An array consisting of x-locations at which to draw vertical lines.
    :return:
    """
    # fig = plt.figure(figsize=(10, 4))
    if mode == "Normal":
        print("Normal distrubution")
        op = NormalTransform(mean=mean, sigma=sigma, key="Normal for Histogram")
    elif mode == "Lognormal":
        op = LognormalTransform(mean=mean, sigma=sigma, key='Lognormal for Histogram', N_copies=0)
    elif mode == "Uniform":
        print("Uniform distribution")
        op = StandardUniformTransform(key='Uniform for Histogram', N_copies=0,
                                      upper_bound=sigma, shift=mean)
    op_samples = np.array([op(s).val for s in [from_random(op.domain) for i in range(n_samples)]])
    label = rf"{mode} with $(\mu, \sigma)=$" + f"$({mean}, {sigma})$" if not (mode=="Uniform") else rf"{mode} in " + r"$\mathrm{[0,1]}$"
    plt.hist(op_samples, bins=200, label=label,
             histtype='step', facecolor='white', color=color)

    if vlines is not None:
        vline_cmb_std = vlines[0]
        vline_sn_std = vlines[1]
        plt.vlines(vline_cmb_std, ymin=0, ymax=350, label=r"$b_{\mathrm{CMB}}$", color="black",
                   ls="--")
        plt.vlines(vline_sn_std, ymin=0, ymax=350, label=r"$b_{\mathrm{SN}}$", color="black")
    plt.ylabel("Frequency", fontsize=30)
    plt.xlabel(r"Fluctuation parameter $b$", fontsize=30)
    # Right-align the text in the legend
    special_legend_II()
    plt.xlim(-0.1, 1.1)
    if save:
        filename = "data_storage/figures/histogram_of_lognormal_distribution"
        plt.tight_layout(pad=2)
        plt.savefig(filename + ".png", pad_inches=1)
    if show:
        plt.show()


def special_legend_II():
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Define the specific replacements
    replacements = {
        r"$a_{\mathrm{CMB}}$": plt.Line2D([0], [0], linestyle="--", color="black", markersize=0),
        r"$a_{\mathrm{SN}}$": plt.Line2D([0], [0], linestyle="-", color="black", markersize=0),
        "End of data": plt.Line2D([0], [0], linestyle="--", color="black", markersize=0),
    }

    # Modify the handles based on the replacements
    handles = [replacements[label] if label in replacements else h for h, label in zip(handles, labels)]

    # Recreate the legend with the modified handles
    plt.legend(handles, labels, fontsize=25)


def special_legend_III():
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Define the specific replacements
    replacements = {
        # "$a_{\mathrm{CMB}}$": plt.Line2D([0], [0], linestyle="--", color="black", markersize=0),
        # "$a_{\mathrm{SN}}$": plt.Line2D([0], [0], linestyle="-", color="black", markersize=0),
        "End of data": plt.Line2D([0], [0], linestyle="--", color=blue, markersize=0),
    }

    # Modify the handles based on the replacements
    handles = [replacements[label] if label in replacements else h for h, label in zip(handles, labels)]

    # Recreate the legend with the modified handles
    plt.legend(handles, labels)


def draw_hubble_diagrams(show=False, save=False, only_show_hist=False):
    z_u, mu_u, covariance_u = read_data_union()
    z_p, mu_p, _ = read_data_pantheon()
    z_d, mu_d, _ = read_data_des()

    convert_to_x = lambda z: np.log(1+z)
    x_u, x_p, x_d = [convert_to_x(z) for z in [z_u, z_p, z_d]]


    min_redshift = np.log(1+0)
    max_redshift = np.log(1+2.26)

    if not only_show_hist:
        plt.subplot(2, 1, 1)

    n_u, bins_u, _ = plt.hist(x_u, bins=10, range=(min_redshift, max_redshift), histtype="step", lw=0, ls="-", color=(0,0,0,0))
    n_p, bins_p, _ = plt.hist(x_p, bins=10, range=(min_redshift, max_redshift), histtype="step", lw=0, ls="", color=(0,0,0,0))
    n_d, bins_d, _ = plt.hist(x_d, bins=10, range=(min_redshift, max_redshift), histtype="step", lw=0, ls="", color=(0,0,0,0))

    # Manually create step-like plot with constant height over each bin
    bin_centers_u = np.repeat(bins_u, 2)[1:-1]
    n_repeated_u = np.repeat(n_u, 2)
    bin_centers_u = np.insert(bin_centers_u, 0, 0)  # Insert 0 at the beginning
    n_repeated_u = np.insert(n_repeated_u, 0, 0)  # Insert 0 at the beginning

    bin_centers_p = np.repeat(bins_p, 2)[1:-1]
    n_repeated_p = np.repeat(n_p, 2)
    bin_centers_p = np.insert(bin_centers_p, 0, 0)  # Insert 0 at the beginning
    n_repeated_p = np.insert(n_repeated_p, 0, 0)  # Insert 0 at the beginning

    bin_centers_d = np.repeat(bins_d, 2)[1:-1]
    n_repeated_d = np.repeat(n_d, 2)
    bin_centers_d = np.insert(bin_centers_d, 0, 0)  # Insert 0 at the beginning
    n_repeated_d = np.insert(n_repeated_d, 0, 0)  # Insert 0 at the beginning

    plt.plot(bin_centers_u, n_repeated_u, linestyle="-", markersize=0, color="black", lw=1,
             label="Union2.1")
    plt.plot(bin_centers_p, n_repeated_p, linestyle="--", markersize=0, color="black", lw=1, dashes=[8, 5],
             label="Pantheon+")
    plt.plot(bin_centers_d, n_repeated_d, linestyle="-.", markersize=0, color="black", lw=1, dashes=[20, 8],
             label="DESY5")
    # dashes for desy5 was: dashes=[20, 15, 1, 1]
    plt.ylabel("Number of SN")
    plt.legend()
    plt.xlim(min_redshift-0.07, max_redshift+0.07)

    if not only_show_hist:
        plt.subplot(2, 1, 2)
        plt.plot(x_u, mu_u, marker="o", lw=0,  markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5,
                 label="Union2.1",)
        plt.plot(x_p, mu_p, marker="s", lw=0, markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5,
                 label="Pantheon+")
        plt.plot(x_d, mu_d, marker="D", lw=0, markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5,
                 label="DESY5")
        plt.ylabel(r"$\mu$")
    plt.xlabel(r"$x=\mathrm{ln}(1+z)$")
    plt.legend()
    plt.xlim(min_redshift-0.07, max_redshift+0.07)

    plt.tight_layout()
    if save:
        plt.savefig("data_storage/figures/hubble_diagram")
    if show:
        plt.show()
    plt.clf()


def plot_h0_comparisons(show=True, save=False):
    value_h0_union = 69.18
    value_h0_union_err = 1.44
    calibr_union = 70
    calibr_union_err = 0

    value_h0_pantheon = 71.40
    value_h0_pantheon_err = 0.81
    calibr_pantheon = 73.04
    calibr_pantheon_err = 1.04

    value_h0_des = 68.31
    value_h0_des_err = 1.36
    calibr_des = 70
    calibr_des_err = 0

    x = ["Union2.1", "Pantheon+", "DESY5"]

    plt.figure(figsize=(10, 16))

    plt.errorbar(x="Union2.1", y=value_h0_union, yerr=value_h0_union_err, fmt="x", color="black", lw=1, ecolor="black",
                 label=r"\texttt{charm2} reconstructed value of $\hat{H}_0$", capsize=4)
    plt.errorbar(x="Union2.1", y=calibr_union, yerr=calibr_union_err, fmt="D", color=blue, lw=1, capsize=4, ecolor=blue,
                 label=r"$\hat{H}_0$ value used in calibration of dataset")

    plt.errorbar(x="Pantheon+", y=value_h0_pantheon, yerr=value_h0_pantheon_err, fmt="x", color="black", lw=1, capsize=4)
    plt.errorbar(x="Pantheon+", y=calibr_pantheon, yerr=calibr_pantheon_err, fmt="D", color=blue, lw=1, capsize=4,
                 ecolor=blue,)

    plt.errorbar(x="DESY5", y=value_h0_des, yerr=value_h0_des_err, fmt="x", color="black", lw=1, capsize=4)
    plt.errorbar(x="DESY5", y=calibr_des, yerr=calibr_des_err, fmt="D", color=blue, lw=1, capsize=4, ecolor=blue,)

    plt.legend(loc="upper right")
    plt.ylabel(r"$H_0$ in units of $\mathrm{km/s/Mpc}$")
    plt.ylim(66, 76)
    if save:
        plt.savefig("data_storage/figures/h0_comparisons.png")
    if show:
        plt.show()
    plt.clf()


def calculate_mode(arr, tolerance=0.1):
    """
    Calculate the mode of an array of discrete floating-point values.

    The mode is determined by clustering values within a specified tolerance
    and finding the cluster with the highest frequency.

    Parameters:
    arr (list of float): The input array of floating-point values.
    tolerance (float, optional): The tolerance within which values are considered
                                 equal for clustering purposes. Default is 0.01.

    Returns:
    float: The mode of the input array, calculated as the average of the values
           in the largest cluster.

    Example:
    >> calculate_mode([0.2, 0.21, 0.22, 0.5, 0.9])
    0.21
    """
    from collections import defaultdict

    clusters = defaultdict(list)

    # Clustering values within the tolerance
    for value in arr:
        placed = False
        for key in clusters:
            if abs(key - value) <= tolerance:
                clusters[key].append(value)
                placed = True
                break
        if not placed:
            clusters[value].append(value)

    # Finding the cluster with the maximum frequency
    mode_cluster = max(clusters.values(), key=len)
    mode_value = sum(mode_cluster) / len(mode_cluster)  # Average of the cluster values

    return mode_value


def pointwise_mode(arrays):
    """
    Calculate the point-wise mode of multiple 1D arrays.

    The mode for each position is determined by clustering values within a
    specified tolerance and finding the cluster with the highest frequency.

    Parameters:
    arrays (list of list of float): A list of 1D arrays containing floating-point values.

    Returns:
    numpy.ndarray: A 1D array containing the point-wise mode of the input arrays.

    Example:
    >> a = [0.2, 0.21, 0.22]
    >> b = [0.2, 0.5, 0.9]
    >> c = [0.2, 0.21, 0.22]
    >> pointwise_mode([a, b, c])
    array([0.2 , 0.21, 0.22])
    """

    # Stack the arrays vertically
    stacked_array = np.vstack(arrays)

    mode_array = []
    for column in stacked_array.T:
        mode_array.append(calculate_mode(column, tolerance=0.1))

    return np.array(mode_array)


def calculate_approximate_mode(posterior_realizations_list, padding_operator, op):
    cropping = padding_operator.adjoint
    posterior_samples = list(posterior_realizations_list.iterator(op))  # Nifty8 Field instances
    posterior_samples_cleaned = [cropping(field).val for field in posterior_samples] # Extracted values
    return pointwise_mode(posterior_samples_cleaned)


def plot_prior_cfm_samples(op, n, x):
    """

    :param op:  The cfm
    :param n:   Number of samples to plot
    :param x:   The field over which to plot the prior samples
    :return:
    """
    samples = list(op(from_random(op.domain)) for _ in range(n))
    fields = [sample.val for sample in samples]
    h = np.array(fields).flatten()
    abs_min, abs_max = (np.min(h), np.max(h))
    for prior_field in samples:
        if max(prior_field.val) < 32.1:
            plt.plot(x.field().val, prior_field.val, lw=2, color=(0,0,0,0.6), markersize=0, ls="-")
    plt.ylim(29.5, 32.2)
    plt.xlabel(r"$x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$")
    plt.ylabel("Prior samples $s(x)$")
    plt.tight_layout()
    plt.savefig("data_storage/figures/PAPER_prior_cfm_samples.png")
    plt.show()


def posterior_parameters(posterior_samples, signal_model, upper_bound_on_fluct = None):
    """
    Finds the posterior statistics for the model parameters from the latent variables xi by getting and analyzing
    the samples returned by the minimize KL function.

    The forward model has fixed distributions for the hyperparameters. The priors on the latent variables xi that
    are propagated to build the fluctuation, loglogavgslope etc., are standard normal Gaussians. But, after the KL
    divergence runs, the xi's have a non-Gaussian posterior distribution, that lead to non-Gaussian posteriors for
    the model parameters. The model parameters are still gotten through the formula

    a = mu + sigma * xi

    But if xi is not a standard normal variable, a will not be a normal distribution.

    From optimize KL you get `posterior_samples`:

        posterior_samples = ift.optimize_kl(likelihood_energy=likelihood_energy,
                                            ... bla bla bla


    These are `MultiFields` that contain the domain and latent (harmonic) xi values for the different model parameters:

        # Make `posterior_samples` iterable:
        poster_samples_list = list(posterior_samples.local_iterator())

        # grab one and print:
        mySample = posterior_samples_list[0]
        print(mySample)

        >> MultiField

        print(mySample.domain)

        >> fluctuations: DomainTuple, len: 0
                line model slope: DomainTuple, len: 0
                line model y-intercept: DomainTuple, len: 0
                loglogavgslope: DomainTuple, len: 0
                xi: DomainTuple, len: 1
                * RGSpace(shape=(8192,), distances=(np.float64(0.41689683388895465),), harmonic=True)

        print(mySample.val)

        >> {'fluctuations': array(0.34721649),
            'line model slope': array(0.21637945),
            'line model y-intercept': array(0.25693888),
            'loglogavgslope': array(-0.82337088),
            'xi': array([ 0.87307418,  2.10056923,  0.21904298, ...,  1.60350815,
                         0.46102179, -3.99240859], shape=(8192,))}


    Now, we get the actual posterior FIELD values by switching from the harmonic to the real domain by doing

        posterior_field_samples = posterior_samples.iterator(s),

    where s is our signal model (e.g. a CFM). The signal model (an operator with an apply method) is applied onto
    each harmonic sample. Somewhere along the operator chain, values for the model parameters must be computed by
    NIFTy.

    We can identify the values by considering that in our specific case:

        loglogavgslope  is from a        normal distribution        and is a scalar
        line slope      is from a        normal distribution        and is a scalar
        line y-offset   is from a        normal distribution        and is a scalar
        fluctuations    is from a        lognormal distribution     and is a scalar
        signal xi_s     is from a        normal distribution        and is an array

    So, IF during the operator chain, the apply method of `LognormalTransform` or `NormalTransform` is called
    five times, it is due to these parameters. In case of `NormalTransform` the last operation done is the
    addition (`Adder` class in `adder.py`) of the mean. In case of `LognormalTransform` it is the
    pointwise exponentiation found in `_FunctionApplier` class in the `operator.py` (iff ` self._funcname` == exp).

    The model parameters can be gotten like this and the user may then guess which parameter is which based on the
    value of the parameter.  ACTUALLY no!! The adder appear 100% according to the order of the parameters in the
    dictionary of the cfm parameters / line model parameters

    The order of the calculated parameter models I THINK is equal to their order as defined in the parameter dictionaries.

    I get the string output of the terminal to do this.

    At the end, you get something like this:

    posterior_parameters_dict = {
    'fluctuations': [
        0.20401470733786994, 0.47579091768502313 ...]}

    IMPORTANT: In order for this function to work the following two print statements are needed:

    File: nifty8 > operators > adder.py
    Print statement: print((x + self._a).val, ";")
    Where: In the apply method of the Adder class directly before returning

    File: nifty8 > operators > operator.py
    Print statement: print("Fluctuations: ", res.val, ";")
    Where: In the apply method of the _FunctionApplier class directly before returning

    File: nifty8 > operators > normal_operators.py
    Edit: Draw log-normal transform with the help of NormalTransform2, which is a copy of NormalTransform that employs
    Adder2 class, a copy of the Adder class without the aforementioned print statement.

    File nifty8 > operators > adder.py:
    Edit: Create a copy of the adder class called `Adder2` that does not contain any print statements.


    :parameter custom_ptw       A string like "exp" or "CDF" which is searched for in the log string and handles
                                the pointwise manipulation of a PDF.


    :return:
    """

    import io
    import contextlib

    def construct_real_space_samples(post_samples, s):
        list(post_samples.iterator(s))  # Nifty8 Field instances

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        construct_real_space_samples(posterior_samples, signal_model)

    output = f.getvalue()
    raw = output.split(";")

    # remove unnecessary line breaks
    for index, el in enumerate(raw):
        raw[index] = el.replace("\n", " ")
        if raw[index] == " ":
            raw.pop(index)

    # remove xi_s lists
    for idx, entry in enumerate(raw):
        if "[" in entry and "]" in entry:
            raw.pop(idx)

    fluctuations_list = []
    # find, store and remove all fluctuation parameters
    for idx, entry in enumerate(raw):
        if "Fluctuations" in entry:
            value = entry.split(":")[1]
            fluctuations_list.append(float(value))
            raw.pop(idx)

    if upper_bound_on_fluct is not None:
        fluctuations_list = [upper_bound_on_fluct * el for el in fluctuations_list]

    loglogavgslope_list = []
    line_slope_list = []
    line_offset_list = []
    # assume the leftover elements are in the order of `loglogavgslope`, `line slope`, `line offset`
    lists = [loglogavgslope_list, line_slope_list, line_offset_list]
    for i, element in enumerate(raw):
        lists[i % 3].append(element)


    # Now turn everything into floats
    loglogavgslope_list = [float(el) for el in loglogavgslope_list]
    line_slope_list = [float(el) for el in line_slope_list]
    line_offset_list = [float(el) for el in line_offset_list]

    # Construct a dictionary:
    posterior_parameters_dict = {
        "fluctuations": fluctuations_list,
        "loglogavgslope": loglogavgslope_list,
        "line slope": line_slope_list,
        "line offset": line_offset_list,
    }

    return posterior_parameters_dict


def visualize_posterior_histograms(posterior_parameters_dict):
    from scipy.stats import norm, lognorm
    # Set up a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # List of parameter names
    parameter_names = list(posterior_parameters_dict.keys())

    for i, param in enumerate(parameter_names):
        # Get the values for the current parameter
        values = posterior_parameters_dict[param]

        # Plot histogram for each parameter
        axes[i].hist(values, bins=20, color='skyblue', edgecolor='black', density=True, label=f"Posterior samples")

        # Overlay the appropriate analytic prior distribution
        if param == "loglogavgslope":
            # Prior: Vertical line at -4 for `loglogavgslope`
            axes[i].axvline(x=-4, color='red', linestyle='--', label="Prior")

        elif param == "fluctuations":
            # Uniform distribution prior
            x = np.linspace(0, 1, 1000)
            pdf = np.ones_like(x) / (max(values) - min(values))
            axes[i].plot(x, pdf, color='black', label="Prior:")
            axes[i].vlines(0.6, 0, 5, linestyle='--', color='red', label="Initial position")
            axes[i].set_xlim(-0.1, 1.1)
            axes[i].set_ylim(0, 6)

        elif param == "line slope":
            # Prior: Normal with mu=2 and sigma=5 for `slope of line`
            mu, sigma = 2, 5
            x = np.linspace(min(values)-5, max(values)+5, 1000)
            pdf = norm.pdf(x, mu, sigma)*np.max(values)
            axes[i].plot(x, pdf, color='black', label="Prior")

        elif param == "line offset":
            # Prior: Normal with mu=30 and sigma=10 for `offset of line`
            mu, sigma = 30, 10
            x = np.linspace(min(values)-10, max(values)+10, 1000)
            pdf = norm.pdf(x, mu, sigma)*np.max(values)
            axes[i].plot(x, pdf, color='black', label="Prior")

        # Add labels and title
        axes[i].set_xlabel(f"Value of {param}")
        axes[i].set_ylabel("Frequency")
        axes[i].legend(loc="upper left")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def construct_initial_position(n_pix_ext, distances, fluctuations):
    """
    Constructs a MultiField that can be fed into `initial_position` of optimize_kl. The MultiField's structure is like
    this:

    domain:
    ---------
    MultiDomain:
      fluctuations: DomainTuple, len: 0
      line model slope: DomainTuple, len: 0
      line model y-intercept: DomainTuple, len: 0
      loglogavgslope: DomainTuple, len: 0
      xi: DomainTuple, len: 1
      * RGSpace(shape=(8192,), distances=(np.float64(0.41671913321298615),), harmonic=True)

    val:
    ---------
    {'fluctuations': array(-0.1528622), 'line model slope': array(0.09607211),
     'line model y-intercept': array(0.21449712), 'loglogavgslope': array(3.96641546e-07),
     'xi': array([ 7.12659478e-08, -4.29667036e-01,  2.03453633e+00, ..., 7.98430340e-01,  1.77930610e-01,
                   1.33588287e-01], shape=(8192,))}

    For the real data inference, we expect the fluctuations to be around ~ 0.14. Indeed,

        cdf(-1.05) = 0.1468590563758959,

    with cdf = scipy.stats.norm.cdf, so we choose xi_fluct = -1.05.

    loglogavgslope in signal space = -4 corresponds to xi_loglog = 3.96641546e-07

    line model slope and offset are set according to the mean xi's of posterior Union2.1 samples.

    The signal xi_s are drawn from a standard random distribution of the length of the extended signal space.

    The domain of the xi_s variable is:

        * RGSpace(shape=(8192,), distances=(np.float64(0.41750518696035904),), harmonic=True)

    (note its in harmonic space so it needs to be the co-space of the regular RG space).

    For scalar values, the domain is a simple scalar domain.

    One last note: Here, i am droing the xi_s values from a simple standard normal distribution.
    But in reality, the xi_s values seem to have some structure. To see this, you may do:

    posterior_samples, final_pos = ift.optimize_kl(return_final_pos = True etc...)
    plt.plot(final_pos.val["xi"])
    plt.show()

    UPDATE: Now I am using the mean xi's array of posterior Union2.1 samples.

    :parameter: n_pix_ext      : The number of points of the extended signal domain.
    :parameter: distances:     : The size of the pixels
    :parameter: fluctuations:  : The wished point-wise std of the field.
    :return:
    """
    import nifty8 as ift

    scalar_domain = ift.DomainTuple.scalar_domain()
    harmonic_RGspace = ift.RGSpace(n_pix_ext, distances=distances).get_default_codomain()

    # xi_values = norm.rvs(size=n_pix_ext)
    xi_values = np.loadtxt("data_storage/raw_data/mean_posterior_union_xi_s.txt")

    xi_fluct = norm.ppf(fluctuations)

    print("\nConstructing initial position... The fluctuation parameter is around ", norm.cdf(xi_fluct))

    fluct = ift.makeField(scalar_domain, arr=np.array(xi_fluct))
    pow_slope = ift.makeField(scalar_domain, arr=np.array(3.96641546e-07))
    line_slope = ift.makeField(scalar_domain, arr=np.array(-0.12847809050927028))
    line_offset = ift.makeField(scalar_domain, arr=np.array(-0.012743781703426963))
    xi_s = ift.makeField(harmonic_RGspace, xi_values)

    init_pos_dict = {"fluctuations": fluct,
                     "line model slope": line_slope,
                     "line model y-intercept": line_offset,
                     "loglogavgslope": pow_slope,
                     "xi": xi_s}

    init_MultiField = ift.MultiField.from_dict(dct = init_pos_dict)

    return init_MultiField


# plot_h0_comparisons(save=True, show=False)
# plot_charm1_in_comparison_fields(save=True, show=True)
# draw_hubble_diagrams(save=True, show=True, only_show_hist=True)
# fig, ax = plt.subplots()
# plot_comparison_fields(plot_fluctuations_scale_visualization=False, ax_object=ax)
# plt.xlabel(r"$x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$", fontsize=30)
# plt.ylabel("$s(x)$", fontsize=30)
# plt.legend(fontsize=25)
# plt.xlim(0, 1.2)
# plt.ylim(29.5, 32.5)
# plt.tight_layout()
# plt.savefig("PAPER_comparison_fields.png")
# plt.show()
ms = [-0.5, ]  # means
ss = [1, ]  # sigmas
colors = ["black", "green"]
for m, s, c in zip(ms, ss, colors):
    plot_lognormal_histogram(mean=m, sigma=s, n_samples=7000, vlines=[0.147, 0.14], save=False, show=False,
                             color=c, mode="Uniform")

plt.ylim(0, 100)
plt.xlim(0, 1)
plt.tight_layout()
# plt.savefig("PAPER_fluctuation_distribution.png")
plt.show()