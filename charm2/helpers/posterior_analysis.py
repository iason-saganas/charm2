from .utilitites import *
import numpy as np
from scipy.constants import G
from typing import Literal

__all__ = ["plot_charm2"]

def _flat_lcdm(x: np.array, H_0, omega_m):

    # Construct the base lcdm, cmb parameters signal field
    m = 3 / (8 * np.pi * G)
    inner_log_func = m * H_0 ** 2 * (1 + omega_m * (np.exp(3 * x) - 1))
    s_base = np.log(inner_log_func)

    return s_base


def _flat_evolving_dark_energy(x: np.array, H_0=68.37, w_a=-8.8, w_0=-0.36, omega_m0=0.495):

    omega_l0 = 1 - omega_m0
    m = 3 / (8 * np.pi * G)
    E_sq = omega_m0*np.exp(3*x) + omega_l0 * np.exp(3*x*(1+w_0+w_a)) * np.exp( -3*(w_a*(1-np.exp(-x))))
    inner_log_func = m * H_0 ** 2 * E_sq
    s_base = np.log(inner_log_func)

    return s_base


def _linear(x, m, t):
    return m * x + t


def _plot_data_domain():
    raise NotImplementedError


def plot_charm2(posterior_samples, LH:_LhContainer, plot_mode:Literal["real", "synthetic"], show=True, save=False,
                plot_domain:Literal["data", "signal", "extended signal"]="signal"):
    print("Reading in data from various compilations for plotting purposes")
    z_p, mu_p, _ = read_data_pantheon()
    z_u, mu_u, _ = read_data_union()
    z_d, mu_d, _ = read_data_des()

    x = LH.meta.x
    s = LH.meta.s_model
    ZP = LH.meta.ZP
    neg_a_mag = LH.meta.neg_a_mag

    # Extract signal posterior samples
    posterior_signal_samples = list(posterior_samples.iterator(s))
    if plot_domain == "extended signal":
        smpls = [field.val.asnumpy() for field in posterior_signal_samples]
    elif plot_domain == "signal":
        smpls = [ZP.adjoint(field).val.asnumpy() for field in posterior_signal_samples]
    elif plot_domain == "data":
        smpls = None
        _plot_data_domain()
    else:
        raise ValueError("plot_domain must be 'signal (extended)' or 'data'")

    s_mean, s_std = np.mean(smpls, axis=0), np.std(smpls, axis=0)

    if plot_mode == "real":
        plot_charm2_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1 + z_u)),
                                         x_max_des=np.max(np.log(1 + z_d)), show=show, save=save, x=x.field().val.asnumpy(),
                                         s=s_mean, s_err=s_std, dataset_used=LH.meta.dataset_name,
                                         neg_a_mag=neg_a_mag, b0=LH.meta.b0, apply_common_labels=True, disable_hist=True,
                                         disable_x_label=False, disable_y_label=False, plot_evolving_dark_energy=False,
                                         # additional_sample_labels="Pantheon 0.2=b0",
                                         # additional_samples=[s_mean_2.val,]
                                         # additional_samples=[
                                         #     flat_lcdm( x.field().val, popt[0][0], popt[0][1] ),
                                         #     flat_evolving_dark_energy(x.field().val, *popt2)]
                                         # ,additional_sample_labels=["flat LCDM fit", "flat evolving dark energy fit"],
                                         # additional_samples=[
                                         #     posterior_line,
                                         #     posterior_cfm
                                         # ]
                                         # ,
                                         # additional_sample_labels=["Posterior line", "Posterior cfm"],
                                         )
    elif plot_mode == "synthetic":

        plot_synthetic_ground_truth(x=x, ground_truth=ZP.adjoint(LH.meta.ground_truth_field).val.asnumpy(),
                                    x_max_pn=np.max(np.log(1 + z_p)),
                                    reconstruction=(s_mean, s_std), save=save,
                                    show=show)
    else:
        raise ValueError("plot_mode must be 'real' or 'synthetic'")
