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

__all__ = ['create_plot_1', 'CustomRGSpace', 'unidirectional_radial_los', 'build_response', 'plot_all_synthetic_pre_kl',
           'kl_sampling_rate', 'plot_all_synthetic_post_kl']


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
        return 50
