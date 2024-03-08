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

__all__ = ['create_plot_1', 'CustomRGSpace', 'unidirectional_radial_los', 'build_response', 'plot_all_synthetic_pre_kl',
           'kl_sampling_rate', 'plot_all_synthetic_post_kl', 'read_data_union', 'read_data_pantheon',
           'CovarianceMatrix', 'raise_warning']


class CustomRGSpace(RGSpace):
    """
    A regular cartesian grid, extended by a class method `field`.

    Topologically, an n-dimensional RGSpace is an n-Torus, i.e. it has periodic
    boundary conditions.

    In `CHARM`, the domain of interest is 0 < x < 1, where x = log(1+z) is called the "negative scale factor magnitude".
    The signal domain length is stretched to 3, so the periodic boundary conditions set by NIFTy do not affect
    the reconstruction in the domain of interest. 3 is equal to log(1+z) where z is approximately redshift 100.

    """
    def __init__(self, shape, distances=None, harmonic=False, _realdistances=None):
        super().__init__(shape, distances=distances, harmonic=harmonic, _realdistances=_realdistances)

    def field(self):
        """
        A regular grid space in `NIFTy` is a geometric object, as a domain for signal fields, has multiple functions.
        This method returns a field of the values associated with the 'distances' property of the raw, geometric object
        `RGSpace`. Example:

        Let the domain of a signal field be a regular, 1D grid space of length 100 and consisting of 5000 pixels.
        Then, each pixel has a width of 0.02.

        Calling this method will return a `NIFTy` field objects whose values are [0.02, 0.04, 0.06, ..., 100].
        This way, the distance values of the domain may be, e.g., added to the signal field, as needed in the signal
        response of `CHARM`.

        This is a custom extension and not part of `NIFTy` source code.


        :return:
        """
        dimension = len(self.shape)
        if dimension != 1:
            raise ValueError(f"Class method 'field' of 'CustomRGSpace' only works for 1D grid spaces, \n"
                             f"not for {dimension}D grid spaces.")

        values = []
        resolution = self.shape[0]
        distances = self.distances[0]
        for iindex in range(1, resolution + 1):
            values.append(iindex * distances)
        dom = DomainTuple.make((self,))
        return Field(dom, val=np.array(values))


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
    ends = np.sort(arr/maximum)
    return ends


def build_response(signal_space: CustomRGSpace, signal: _OpChain, data_space: UnstructuredDomain,
                   neg_scale_factor_mag: np.array) -> _OpChain:
    """
    Builds the `CHARM` signal response as a chain of operators.
    :param signal_space:            'CustomRGSpace',        Number of lines-of-sight to construct.
    :param signal:                  '_OpChain',             The correlated field modelling the signal whose target is
                                                            'signal_space'.
    :param data_space:              'UnstructuredDomain',   The data space.
    :param neg_scale_factor_mag:    np.array,               The measured negative scale factor mags.
    :return: R: _OpChain, The signal response.
    """
    s = signal
    x = signal_space
    neg_a_mag = neg_scale_factor_mag
    K = (8 * np.pi * G / 3) ** (-1 / 2) * c
    n_dp = len(neg_a_mag)

    los_integration = LOSResponse(domain=x, starts=[np.zeros(int(n_dp))], ends=[neg_a_mag])
    offset_by_x_tilde = Adder(a=x.field(), neg=False)
    e_to_the_power_of_x = DiagonalOperator(diagonal=Field.from_raw(data_space, np.exp(neg_a_mag)))
    offset_by_negative_five = Adder(a=5, neg=True, domain=DomainTuple.make(data_space, ))

    R = offset_by_negative_five(5 * np.log10(e_to_the_power_of_x(K * los_integration(
        np.exp(offset_by_x_tilde(-1 / 2 * s))))))
    return R


def create_plot_1(x, y, color, x_label, y_label, title, x_lim=None, y_lim=None, ls=".", legend_label=None):
    """
    Creates a standard plot.
    """
    if x is None:
        # Drawing log scale power spectrum against log k modes
        x = np.log(range(len(y)+1)[1:len(y)+1])  # Cut out first element, which is zero for some reason and shift 1
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
                  y_label='Signal Field $s(x)$', title='Signal Field Ground Truth', y_lim=(-1.5, 1))

    create_plot_1(x=None, y=log_power_spectrum, color='black',
                  x_label=r'Fourier Modes $\mathrm{log}(k)$', y_label=r'Power Spectrum $\mathrm{log}((p_s(k))$',
                  title='Power Spectrum Ground Truth', x_lim=(0, 9), y_lim=(-30, 5), ls='.')

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
    plt.xlim(0, 1)
    plt.ylim(-0.8, 3)
    plt.plot(neg_a_mag, 2 * np.ones(len(neg_a_mag)), ".", lw=0, color="black", label='Distribution Of Datapoints')
    plt.legend()
    plt.show()

    plt.errorbar(x_field_val, s_mean_val-ground_truth_signal_val, yerr=sqrt_s_var_val, fmt='-', color=blue,
                 ecolor=light_blue, label=r'$1\sigma$')
    plt.errorbar(x_field_val, s_mean_val - ground_truth_signal_val, color=blue, yerr=2*sqrt_s_var_val, fmt='-',
                 ecolor=lighter_blue, label=r'$2\sigma$')
    plt.errorbar(x_field_val, s_mean_val - ground_truth_signal_val, color=blue, yerr=3 * sqrt_s_var_val, fmt='-',
                 ecolor=lightest_blue, label=r'$3\sigma$')
    plt.xlabel(r'Negative Scale Factor Magnitude $x=-\mathrm{log}(a)=\mathrm{log}(1+z)$')
    plt.ylabel(r'Residuals $\Delta s(x)$')
    plt.title('Reconstructed Signal Field Deviation From Ground Truth')
    plt.plot(neg_a_mag, 0.75 * np.ones(len(neg_a_mag)), ".", lw=0, color=light_red, label='Distribution Of Datapoints')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(-0.5, 1)
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

    """
    def __init__(self, domain, matrix, tol, spaces=None, flatten=False, sampling_dtype=None):
        super().__init__(domain, matrix, spaces=spaces, flatten=flatten)
        self._dtype = sampling_dtype
        self.tol = tol
        self.mtrx_sqrt = self.get_fct()

    @property
    def inverse(self):
        # Return N^{-1}, such that N^{-1} @ N ~ 1. The off-diagonal elements are of the order 10e-18 -> 10e-20.
        return np.linalg.inv(self._mat)

    def get_fct(self):
        print("get_fct called. Should only appear once.")
        """
        Let self._mat = N.
        Let N.inverse = N^{-1} be a strictly positive, symmetric matrix. Then, it has a unique matrix square root via
        the spectral theorem:

            √N^{-1} = U √D U^†,

        where √D:=diag(√λ_1, √λ_2, ..., √λ_n). This method calls the function 'spectral_decompose' (which checks for
        positivity and symmetry) and returns the inverse covariance matrix square root √N^{-1}.
        """
        N_inv = self.inverse
        U, D, U_inv = spectral_decompose(N_inv, tol=self.tol)
        square_root = U @ np.sqrt(D) @ U_inv
        sanity_check = np.allclose(N_inv, square_root @ square_root, atol=self.tol, rtol=0)
        if not sanity_check:
            raise ValueError("Inverse covariance matrix is not approximately equal to the square of its"
                             "square root decomposition.")
        return square_root

    def get_sqrt(self):
        """
        Returns the matrix square root √N^{-1} as new 'MatrixProductOperator'.
        """
        return MatrixProductOperator(self._domain, self.mtrx_sqrt)


def raise_warning(message):
    warnings.warn(message, category=Warning)

