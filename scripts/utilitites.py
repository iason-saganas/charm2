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
from nifty8.operators.normal_operators import NormalTransform, LognormalTransform
import matplotlib.pyplot as plt
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

__all__ = ['create_plot_1', 'unidirectional_radial_los', 'build_response', 'kl_sampling_rate', 'read_data_union',
           'read_data_pantheon', 'CovarianceMatrix', 'raise_warning', 'build_flat_lcdm', 'pickle_me_this',
           'unpickle_me_this', 'current_expansion_rate', 'attach_custom_field_method', 'chi_square',
           'build_charm1_agnostic', 'plot_comparison_fields', 'show_plot', 'plot_flat_lcdm_fields',
           'plot_charm1_in_comparison_fields', 'LineModel', 'plot_synthetic_ground_truth', 'plot_synthetic_data',
           'PiecewiseLinear', 'plot_charm2_in_comparison_fields', 'plot_prior_distribution', 'LCDM_MODEL',
           'read_data_des', 'store_meta_data']


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
    alpha = contraction.adjoint @ alpha
    beta = contraction.adjoint @ beta
    x_coord = DiagonalOperator(diagonal=x.field())
    line_model = x_coord @ alpha + beta
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
        arr = 2*np.random.rand(n_los)
        ends = np.sort(arr)
    else:
        arr = np.random.lognormal(mean=0, sigma=1, size=n_los)
        maximum = np.max(arr)
        ends = np.sort(arr / maximum)
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
    plt.xlabel(r'Negative Scale Factor Magnitude $x=-\mathrm{log}(a)=\mathrm{log}(1+z)$')
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
    plt.xlabel(r'Negative Scale Factor Magnitude $x=-\mathrm{log}(a)=\mathrm{log}(1+z)$')
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
        path_to_x = "../data_storage/raw_data/x_field_charm1_union2_1.txt"
        path_to_s = "../data_storage/raw_data/s_field_charm1_union2_1.txt"
        path_to_D = r"../data_storage/raw_data/D_field_charm1_union2_1.txt"
        correction_offset = 0.0246747  # log(rho/rho0) in charm1 is not precisely at 0, which must be incorrect
    elif mode == "pantheon+":
        path_to_x = "../data_storage/raw_data/x_field_charm1_pantheon+.txt"
        path_to_s = "../data_storage/raw_data/s_field_charm1_pantheon+.txt"
        path_to_D = r"../data_storage/raw_data/D_field_charm1_pantheon+.txt"
        correction_offset = 0.10272  # log(rho/rho0) in charm1 is not precisely at 0, which must be incorrect
    elif mode == "pantheon+_reformulated":
        path_to_x = "../data_storage/raw_data/charm1_reformulated/x_field_charm1_pantheon+.txt"
        path_to_s = "../data_storage/raw_data/charm1_reformulated/s_field_charm1_pantheon+.txt"
        path_to_D = r"../data_storage/raw_data/charm1_reformulated/D_field_charm1_pantheon+.txt"
    else:
        raise ValueError("Unrecognized mode in `build_charm1_agnostic`.")

    if mode != "pantheon+_reformulated":
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


def store_meta_data(datetime_obj, duration_of_inference, len_d, inference_type):
    """
    Stores metadata related to an inference run in a text file and manages associated data files.

    This function creates a metadata file named using the current date and time. The file contains
    details about the inference run, including the type of data used, the length of the dataset, and
    the duration of the inference in minutes. The function also appends the content of additional files
    located in a temporary directory to the metadata file, then moves the metadata file to a specified
    directory and deletes the temporary directory.

    Args:
        datetime_obj (datetime object): The datetime object representing when the inference run occurred.
        duration_of_inference (float): The duration of the inference in seconds.
        len_d (int): The length of the dataset used for the inference.
        inference_type (str): A string indicating the type of inference ('synthetic' or 'real').

    """
    # Create file name with datetime
    timestamp = datetime_obj.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"metadata_{timestamp}.txt"

    # Convert duration from seconds to minutes
    duration_minutes = duration_of_inference / 60

    # Define paths
    temp_dir = 'data_storage/pickled_inferences/temp'
    final_dir = f'data_storage/pickled_inferences/{inference_type}'

    # Ensure final directory exists
    os.makedirs(final_dir, exist_ok=True)

    # Create the metadata file
    with open(filename, 'w') as file:
        file.write(f"Charm2 inference run on {timestamp}. Mode: {inference_type}\n")
        file.write(f"Length of dataset: {len_d}\n")
        file.write(f"Time took in minutes: {duration_minutes:.2f}\n\n")

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
    shutil.rmtree(temp_dir)


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


def chi_square(vector1, vector2):
    # Compute squared differences
    squared_diff = (vector1 - vector2) ** 2

    # Divide by vector1 and sum
    chi_square_val = np.sum(squared_diff / vector1)

    return chi_square_val


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


def plot_comparison_fields(plot_fluctuations_scale_visualization=False):
    """
    Adds comparison fields to a plot, but does not show it.

    :param plot_fluctuations_scale_visualization: bool, if true, adds and shows a plot of linear fits to the comparison
    signal fields, as well as the residuals between said fits and the actual curves.
    This represents the point-wise fluctuations around the offset the correlated field needs to model.
    We take thus take the fluctuation parameter to be the square root of the mean squared residuals.

    :return: handles: tuple,    A tuple containing three strings, representing x label, y label and title for the plot
                                that can be fed into the function `show_plot()`.
    """
    x_coordinates, cmb, sn = build_comparison_fields()
    x, x_sparse = x_coordinates
    s_cmb, s_cmb_err, s_cmb_sparse, H0_cmb = cmb
    s_sn, s_sn_err, s_sn_sparse, H0_sn = sn
    plt.errorbar(x=x, y=s_cmb, yerr=s_cmb_err, fmt="None",
                 ecolor=(0, 0, 0, 0.1))
    plt.errorbar(x=x, y=s_sn, yerr=s_sn_err, fmt="None",
                 ecolor=(0, 0, 0, 0.1))
    dash_dot_dotted = (0, (3, 5, 1, 5, 1, 5))
    plt.plot(x_sparse, s_cmb_sparse, ls=dash_dot_dotted, lw="1", color="black",
             label=r'$s_{\mathrm{CMB}}$. $\hat{H}_0=' + str(H0_cmb) + '$',
             )
    long_dash_with_offset = (5, (10, 3))
    plt.plot(x_sparse, s_sn_sparse, ls=long_dash_with_offset, lw="1", color="black",
             label=r'$s_{\mathrm{SN}}$. $\hat{H}_0=' + str(H0_sn) + '$',
             )
    t = r"Flat $\Lambda$CDM Signal Fields." + "\nComparison Between CMB And Supernovae Measurements."
    xl = r"$x=-\mathrm{log}(a)=\mathrm{log}(1+z)$"
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
              y_label: str = "", ):
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
    plt.legend()
    if x_lim is not None:
        plt.xlim(*x_lim)
    if y_lim is not None:
        plt.ylim(*y_lim)
    if title != "":
        plt.title(title)
    if x_label != "":
        plt.xlabel(x_label)
    if y_label != "":
        plt.ylabel(y_label)
    if save_filename != "":
        plt.tight_layout(pad=2)
        plt.savefig(save_filename + ".png", pad_inches=1)
    if show:
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
                                     show: bool = False, save: bool = True, ):
    """
    Plots the reconstructed charm1 curve using Union2.1 data into a figure containing CMB and SN comparison fields.
    :param s_err:           The charm2 posterior standard deviation values.
    :param s:               The charm2 posterior mean values.
    :param x:               The coordinate axis of the charm2 reconstruction.
    :param save:
    :param x_max_union:     A vertical line is plotted at this point, indicating the end of the dataset.
    :param x_max_pn:        The max scale factor magnitude of the pantheon analysis.
    :param show:
    :return:
    """
    xl, yl, t = plot_comparison_fields()
    plt.vlines(x_max_union, 0, 50, linestyles='dashed', label="End of Union2.1 data")
    current_expansion_mean, current_expansion_err = current_expansion_rate(s, s_err)
    h0_charm2 = str(current_expansion_mean)
    print("Calculated value of H0: ", current_expansion_mean, " ± ", current_expansion_err)
    if save:
        filename = "data_storage/figures/charm2_reconstruction_pantheon+_cfm_model_fluct_0_2_0_2_final_maybe"
    else:
        filename = ""
    plt.errorbar(x=x, y=s, yerr=s_err, color=blue, ecolor=light_blue, label=r"\texttt{CHARM2},"
                                                                            r"Pantheon+ data. $\hat{H}_0=" + h0_charm2
                                                                            + "$",
                 markersize=1)
    show_plot(x_lim=(0, x_max_pn), y_lim=(29.5, 32.5), x_label=xl, y_label=yl, title="",
              save_filename=filename, show=show)


def plot_charm1_in_comparison_fields(x_max_pn: float, x_max_union: float, show: bool = False, save: bool = True):
    """
    Plots the reconstructed charm1 curve using Union2.1 or Pantheon+ data into a figure containing CMB and SN
    comparison fields.
    :param save:
    :param x_max_union:     A vertical line is plotted at this point, indicating the end of the dataset.
    :param x_max_pn:        The max scale factor magnitude of the pantheon analysis.
    :param show:
    :return:
    """
    xl, yl, t = plot_comparison_fields()
    plt.vlines(x_max_union, 0, 50, linestyles='dashed', label="End of Union2.1 data")
    x, s, s_err = build_charm1_agnostic(mode="union2.1")
    h0_charm1 = str(current_expansion_rate(s))
    if save:
        filename = "data_storage/figures/charm1_reformulated_reconstruction_with_comparison_fields_pantheon+"
    else:
        filename = ""
    plt.errorbar(x=x, y=s, yerr=s_err, color=blue, ecolor=light_blue, label=r"\texttt{CHARM1},"
                                                                            r"Pantheon+ data. $\hat{H}_0=" + h0_charm1 + "$",
                 markersize=1)
    show_plot(x_lim=(0, x_max_pn), y_lim=(29.5, 32.5), x_label=xl, y_label=yl, title="",
              save_filename=filename, show=show)


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
    plt.vlines(2, 32, 37, linestyles='dashed', label="End of synthetic data", color="black")
    xl = r"$x=-\mathrm{log}(a)=\mathrm{log}(1+z)$"
    yl = r"$s(x)$"
    if save:
        filename = "data_storage/figures/synthetic_ground_truth_with_reconstruction"
    else:
        filename = ""
    plt.plot(x, ground_truth, "-", color="black", lw=1, label="Synthetic ground truth")
    if reconstruction is not None:
        plt.errorbar(x=x, y=reconstruction[0], yerr=reconstruction[1], color=blue, ecolor=light_blue,
                     label=r"$\texttt{CHARM2}$ synthetic reconstruction", markersize=1)
    show_plot(x_lim=(0, 2), y_lim=(32, 40.5), x_label=xl, y_label=yl, title="",
              save_filename=filename, show=show)


def plot_synthetic_data(neg_scale_fac_mag: np.ndarray, data: np.ndarray, x_max_pn: float, mu_array: np.array,
                        show=True, save=True):
    """
    Plots the synthetic ground truth and data it creates.
    :param show: bool,              Whether to show the plot
    :param save: bool,              Whether to save the plot
    :param mu_array: np.array,     The mock distance moduli
    :param x_max_pn: float,     The max scale factor magnitude of the pantheon analysis.
    :param neg_scale_fac_mag:   The unidirectional line of sights x=np.log(1+z)
    :param data:                The constructed synthetic data
    :return:
    """
    mu_min = np.min(mu_array)
    mu_max = np.max(mu_array)

    x_min = np.min(neg_scale_fac_mag)
    x_max = np.max(neg_scale_fac_mag)

    xl = r"$x=-\mathrm{log}(a)=\mathrm{log}(1+z)$"
    yl = r"$\mu (x)$"
    if save:
        filename = "data_storage/figures/synthetic_data"
    else:
        filename = ""

    plt.subplot(2, 1, 1)
    plt.hist(neg_scale_fac_mag, histtype="step", color="black", lw=0.5, bins=10)
    plt.ylabel(r"$\#$ of datapoints")
    plt.xlim(-0.1+x_min, x_max+0.1)
    plt.subplot(2, 1, 2)
    plt.plot(neg_scale_fac_mag, data, ".", color="black", label="Synthetic data points")
    show_plot(x_lim=(-0.1+x_min, x_max+0.1), y_lim=(mu_min-0.1, mu_max+0.1), x_label=xl, y_label=yl, title="",
              save_filename=filename, show=show)


def PiecewiseLinear(signal_space: RGSpace, omega_m_custom: float = None, omega_l_custom: float = None):
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
    x_coord = DiagonalOperator(diagonal=Field(domain=DomainTuple.make(x), val=np.exp(5 * x.field().val)))

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


def LCDM_MODEL(signal_space):
    """
    Flat LCDM
    """
    x = signal_space

    expander = ContractionOperator(domain=x, spaces=0).adjoint
    omega_m = expander @ LognormalTransform(.3, .1, key="Omega m", N_copies=0)
    rho_0 = expander @ LognormalTransform(8269771251557, 8269771251557 / 10, key="rho 0", N_copies=0)

    import nifty8 as ift
    e_to_the_power_of_3x_field = (x.field() * 3).exp() - 1
    e_to_the_pow_op = DiagonalOperator(diagonal=e_to_the_power_of_3x_field)

    # one = DiagonalOperator(diagonal=ift.makeField(domain=x, arr=np.ones(x.shape[0])))
    one = ift.Adder(a=ift.makeField(domain=x, arr=np.ones(x.shape[0])))
    part1 = e_to_the_pow_op @ omega_m
    one_plus_part1 = one @ part1
    logi = one_plus_part1.log()

    return logi + rho_0.log()


def plot_lognormal_histogram(mean: float, sigma: float, n_samples: int, vlines: np.array = None, save=False, show=True):
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
    op = LognormalTransform(mean=mean, sigma=sigma, key='Lognormal for Histogram', N_copies=0)
    op_samples = np.array([op(s).val for s in [from_random(op.domain) for i in range(n_samples)]])
    plt.hist(op_samples, bins=200, label=r"Lognormal with $(\mu, \sigma)=$" + f"$({mean}, {sigma})$",
             histtype='step', facecolor='white', color="black")

    if vlines is not None:
        vline_cmb_std = vlines[0]
        vline_sn_std = vlines[1]
        plt.vlines(vline_cmb_std, ymin=0, ymax=350, label=r"$a_{\mathrm{CMB}}$", color="black",
                   ls="--")
        plt.vlines(vline_sn_std, ymin=0, ymax=350, label=r"$a_{\mathrm{SN}}$", color="black")
    plt.ylabel("Samples")
    plt.xlabel("Bins")
    # Right-align the text in the legend
    plt.legend()
    plt.xlim(0, 1)
    if save:
        filename = "data_storage/figures/histogram_of_lognormal_distribution"
        plt.tight_layout(pad=2)
        plt.savefig(filename + ".png", pad_inches=1)
    if show:
        plt.show()


def draw_hubble_diagrams(show=False, save=False):
    z_u, mu_u, covariance_u = read_data_union()
    z_p, mu_p, _ = read_data_pantheon()
    z_d, mu_d, _ = read_data_des()

    convert_to_x = lambda z: np.log(1+z)
    x_u, x_p, x_d = [convert_to_x(z) for z in [z_u, z_p, z_d]]

    plt.subplot(2, 1, 1)

    min_redshift = np.log(1+0)
    max_redshift = np.log(1+2.26)

    n_u, bins_u, _ = plt.hist(x_u, bins=10, range=(min_redshift, max_redshift), histtype="step", lw=0, ls="-")
    n_p, bins_p, _ = plt.hist(x_p, bins=10, range=(min_redshift, max_redshift), histtype="step", lw=0, ls="")
    n_d, bins_d, _ = plt.hist(x_d, bins=10, range=(min_redshift, max_redshift), histtype="step", lw=0, ls="")

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

    plt.plot(bin_centers_u, n_repeated_u, linestyle="-", markersize=0, color="black", lw=0.5,
             label="Union2.1")
    plt.plot(bin_centers_p, n_repeated_p, linestyle="--", markersize=0, color="black", lw=0.5, dashes=[10, 5],
             label="Pantheon+")
    plt.plot(bin_centers_d, n_repeated_d, linestyle="-.", markersize=0, color="black", lw=0.5, dashes=[20, 15, 1, 1],
             label="DESY5")
    plt.ylabel("Number of SN")
    plt.legend()
    plt.xlim(min_redshift-0.07, max_redshift+0.07)

    plt.subplot(2, 1, 2)
    plt.plot(x_u, mu_u, marker="o", lw=0,  markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5,
             label="Union2.1",)
    plt.plot(x_p, mu_p, marker="s", lw=0, markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5,
             label="Pantheon+")
    plt.plot(x_d, mu_d, marker="D", lw=0, markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5,
             label="DESY5")
    plt.xlabel(r"$x=\mathrm{log}(1+z)$")
    plt.ylabel(r"$\mu$")
    plt.legend()
    plt.xlim(min_redshift-0.07, max_redshift+0.07)

    plt.tight_layout()
    if save:
        plt.savefig("data_storage/figures/hubble_diagram")
    if show:
        plt.show()
    plt.clf()

# draw_hubble_diagrams(save=True)
# plot_comparison_fields(plot_fluctuations_scale_visualization=True)
# plot_lognormal_histogram(mean=.4, sigma=.2, n_samples=10000, vlines=[0.147, 0.14], save=True, show=True)
