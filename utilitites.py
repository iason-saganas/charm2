from nifty8.domains.rg_space import RGSpace
from nifty8.domain_tuple import DomainTuple
from nifty8.field import Field
import numpy as np
from scipy.constants import c, G
from typing import Union, Tuple, List, Any
from nifty8.operators.diagonal_operator import DiagonalOperator
from nifty8.operators.adder import Adder
from nifty8.library.los_response import LOSResponse
from nifty8.operators.operator import _OpChain
from nifty8.domains.unstructured_domain import UnstructuredDomain
import matplotlib.pyplot as plt
from data_storage.style_components.matplotlib_style import *
import pandas as pd
from nifty8.operators.matrix_product_operator import MatrixProductOperator
import warnings
import pickle

__all__ = ['create_plot_1', 'unidirectional_radial_los', 'build_response', 'plot_all_synthetic_pre_kl',
           'kl_sampling_rate', 'plot_all_synthetic_post_kl', 'read_data_union', 'read_data_pantheon',
           'CovarianceMatrix', 'raise_warning', 'build_lcdm_with_cmb_parameters', 'pickle_me_this', 'unpickle_me_this',
           'current_expansion_rate_experimental', 'attach_custom_field_method']


def attach_custom_field_method(space: RGSpace):
    """
    Takes an instance of `RGSpace`, a regular cartesian grid, and attaches a new method called `field` which can then
    be called on the `RGSpace` instance.
    The `field` method returns a field of the values associated with the `distances` property of the raw,
    geometric object `RGSpace`.
    Example:

    Let the domain of a signal field be a regular, 1D grid space of length 100 and consisting of 5000 pixels.
    Then, each pixel has a width of 0.02.

    Calling this method will return a `NIFTy` field object whose values are [0.02, 0.04, 0.06, ..., 100].
    This way, the distance values of the domain may be, e.g., added to the signal field, as needed in the signal
    response of `CHARM`.
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
        for iindex in range(1, resolution + 1):
            values.append(iindex * distances)
        dom = DomainTuple.make((rg_space,))
        return Field(dom, val=np.array(values))

    space.field = lambda: field(space)
    return space


def unidirectional_radial_los(n_los: int) -> np.ndarray:
    """
    Generates an ordered array of normally distributed random numbers. Returns that array ('ends') and the 'starts'
    array (zeroes). Represents synthetic redshifts.
    :param n_los:   int,                                    Number of lines-of-sight to construct.
    :return: ends: np.array,                                The synthetic lines-of-sight.
    """
    n_los = int(n_los)
    arr = np.random.lognormal(mean=0, sigma=1, size=n_los)
    maximum = np.max(arr)
    ends = np.sort(arr / maximum)
    return ends


def build_response(signal_space: RGSpace, signal: _OpChain, data_space: UnstructuredDomain,
                   neg_scale_factor_mag: np.array) -> _OpChain:
    """
    Builds the `CHARM` signal response as a chain of operators.
    :param signal_space:            'RGSpace',        Number of lines-of-sight to construct.
    :param signal:                  '_OpChain',             The correlated field modelling the signal whose target is
                                                            'signal_space'.
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
    print("e_to_the_power_of_x: ",e_to_the_power_of_x.val)
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


def plot_all_synthetic_pre_kl(x_field_val, ground_truth_signal_val, log_power_spectrum, synthetic_data_val,
                              neg_scale_factor_mags):
    create_plot_1(x=x_field_val, y=ground_truth_signal_val, color='black', ls='-',
                  x_label=r'Negative Scale Factor Magnitude $x=-\mathrm{log}(a)=\mathrm{log}(1+z)$',
                  y_label='Signal Field $s(x)$', title='Signal Field Ground Truth')

    create_plot_1(x=None, y=log_power_spectrum, color='black',
                  x_label=r'Fourier Modes $\mathrm{log}(k)$', y_label=r'Power Spectrum $\mathrm{log}((p_s(k))$',
                  title='Power Spectrum Ground Truth', ls='.')

    create_plot_1(x=neg_scale_factor_mags, y=synthetic_data_val, color='black',
                  x_label=r'Negative Scale Factor Magnitude $x=-\mathrm{log}(a)=\mathrm{log}(1+z)$',
                  y_label=r'Distance modulus $\mu(x)$', title='Synthetic Data')


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
    if index < 5:
        return 5
    else:
        return 15


def read_data_union():
    path_to_data = 'data_storage/raw_data/Union_2_1_data.txt'
    path_to_cov = 'data_storage/raw_data/Union2_1_Cov-syst.txt'
    z = np.loadtxt(path_to_data, usecols=[1], dtype=float)  # read data
    sorting_indices = np.argsort(z)  # get sorting indices (from smallest to biggest), s.t. mu can be sorted in that
    # order as well
    z = z[sorting_indices]  # sort from smallest to biggest
    mu = np.loadtxt(path_to_data, usecols=[2], dtype=float)[sorting_indices]
    covariance = np.loadtxt(path_to_cov, dtype=float)
    return z, mu, covariance


def read_data_pantheon():
    path_to_data = 'data_storage/raw_data/Pantheon+SH0ES.csv'
    path_to_cov = 'data_storage/raw_data/Pantheon+SH0ES_STAT+SYS.txt'

    required_fields = ["zHEL", "MU_SH0ES"]
    df = pd.read_csv(path_to_data, sep=" ", skipinitialspace=True, usecols=required_fields)
    z = np.array(df.zHEL)
    mu = np.array(df.MU_SH0ES)

    df = pd.read_table(path_to_cov)
    arr = np.array(df["1701"])  # first line corresponds to '1701' = length of data array
    covariance = arr.reshape(1701, 1701)

    return z, mu, covariance


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


def raise_warning(message):
    warnings.warn(message, category=Warning)


def build_lcdm_with_cmb_parameters(x):
    """
    Reference: https://arxiv.org/pdf/1807.06209.pdf, equations (13), (14) and (15).
    A flat universe is assumed, k=0 and the radiation density today is well approximated by 0.
    This function returns the field values as well as the 1σ uncertainty of this base signal field:

    s_base(x) = log( m * hat{H_0}^2 * (Ω_Λ + Ω_m e^(3x) ) ),

    where m is a constant, m := 3/(8π) * (G/[G])^(-1) and hat{H_0} is H_0/(km/s/Mpc). The uncertainty is a function of
    x itself and is calculated via the Gaussian Propagated Error:

    Δs_base(x) = (Σ_i (( ∂ s_base(x) / ∂ x ) Δx ) ^2 )^(1/2),

    where the sum runs over all uncertain values, i.e. Ω_Λ, Ω_m and H_0.
    ToDo: Attention! The formula for Δs_base(x) only holds if the errors are independent of each other, which I am
     pretty sure is not the case for any of Ω_Λ, Ω_m and H_0.

    The values used are:

    H_0 = 67.36 ±  0.54
    Ω_m = 0.3166 ± 0.0084
    Ω_Λ = 0.6847 ± 0.0073

    """
    H_0 = 67.36
    delta_H_0 = 0.54

    omega_m = 0.3166
    delta_omega_m = 0.0084

    omega_l = 0.6847
    delta_omega_l = 0.0073

    # Construct the base lcdm, cmb parameters signal field
    m = 3 / (8 * np.pi * G)
    s_base = np.log(m * H_0 ** 2 * (omega_l + omega_m * np.exp(3 * x)))

    # Construct its 1σ uncertainty array
    common_denominator = (m * H_0 ** 2 * (omega_l + omega_m * np.exp(3 * x))) ** (-1)
    partial_omega_m = common_denominator * m * H_0 ** 2 * np.exp(3 * x)
    partial_omega_l = common_denominator * m * H_0 ** 2
    partial_H_0 = common_denominator * 2 * m * H_0 * (omega_l + omega_m * np.exp(3 * x))

    # Order: First Ω_m, then Ω_Λ, then H0
    all_errors = [delta_omega_m, delta_omega_l, delta_H_0]
    all_derivatives = [partial_omega_m, partial_omega_l, partial_H_0]

    gaussian_error_to_sum = np.zeros(len(x))
    for der, er in zip(all_derivatives, all_errors):
        gaussian_error_to_sum += (der * er) ** 2

    gaussian_error = np.sqrt(gaussian_error_to_sum)

    return s_base, gaussian_error


def pickle_me_this(filename: str, data_to_pickle: object):
    path = "data_storage/pickled_inferences/real/" + filename + ".pickle"
    file = open(path, 'wb')
    pickle.dump(data_to_pickle, file)
    file.close()


def unpickle_me_this(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def current_expansion_rate_experimental(s: np.array) -> float:
    """
    The relationship between the expansion rate and the signal is:

    H(s)^2 / (km/s/Mpc)^2 = 8πG/[G] * e^s(x),

    in other words the Hubble constant H0 = H(s0) = H(s(0)) is

    H0 / (km/s/Mpc)  = sqrt( 8πG/[G] * e^s0).

    ToDo: s0 = s[0] is only APPROXIMATELY the y-intercept of s, because x does
     not start at 0 (or does it?). Fix.

    The returned value is H0 in km/s/Mpc rounded to two decimal places.

    """
    s0 = s[0]
    return np.round(np.sqrt(8 * np.pi * G / 3 * np.exp(s0)), 2)
