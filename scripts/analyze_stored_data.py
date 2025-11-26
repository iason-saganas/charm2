from data_storage.style_components.matplotlib_style import *
# Run from root directory as: `python3.12 -m scripts.analyze_stored_data`
from scipy.constants import G, c

cosmological = True
extend = False
data_to_analyze = "Union2.1"
plot_in_signal_space = True
plot_in_data_space = False
if cosmological:
    from .CONFIG_cosmological import *
    likelihood, d, neg_a_mag, arguments, x, X, s, init_pos, cov = cosmological_likelihood(data_to_use=data_to_analyze)
    data_space = d.domain
else:
    from .CONFIG_synthetic import *
    raise ValueError("not implemented yet")

# fluct_range = np.arange(0.1, 1, 0.1)
# fluct_range = np.linspace(.05, 0.3, 5)  # for DESY5 low fluctuations experiment
fluct_range = [0.05]

corresponding_pickles = [
    #"des.pickle"  # for fluct_init = 0.6
    # "pantheon_overfitting_1.pickle",
    # "pantheon_overfitting_3.pickle",
    # "PAPER_b0_0_6_pantheon.pickle"
    "not in list"
]

# corresponding_pickles = ["2025-05-25_13-34-37_init_fluct_is_0.05.pickle",
# "2025-05-25_15-25-35_init_fluct_is_0.1125.pickle",
# "2025-05-25_17-17-51_init_fluct_is_0.175.pickle",
# "2025-05-25_19-10-09_init_fluct_is_0.2375.pickle",
# "2025-05-25_21-00-17_init_fluct_is_0.3.pickle"
# ]

def flat_lcdm(x: np.array, H_0, omega_m):

    # Construct the base lcdm, cmb parameters signal field
    m = 3 / (8 * np.pi * G)
    inner_log_func = m * H_0 ** 2 * (1 + omega_m * (np.exp(3 * x) - 1))
    s_base = np.log(inner_log_func)

    return s_base


def flat_evolving_dark_energy(x: np.array, H_0=68.37, w_a=-8.8, w_0=-0.36, omega_m0=0.495):

    omega_l0 = 1 - omega_m0
    m = 3 / (8 * np.pi * G)
    E_sq = omega_m0*np.exp(3*x) + omega_l0 * np.exp(3*x*(1+w_0+w_a)) * np.exp( -3*(w_a*(1-np.exp(-x))))
    inner_log_func = m * H_0 ** 2 * E_sq
    s_base = np.log(inner_log_func)

    return s_base


def linear(x, m, t):
    return m * x + t


for idx, fluct in enumerate(fluct_range):

    print("Fluctuations ", fluct)

    # pickle = corresponding_pickles[idx]

    # samples = unpickle_me_this("real/further_analysis/desy5_3.pickle", absolute_path=False)
    # samples = unpickle_me_this(f"/Users/iason/PycharmProjects/Charm2/data_storage/desy5_low_fluct/all_pickles/{pickle}", absolute_path=True)
    samples = unpickle_me_this("/Users/iason/PycharmProjects/Charm2/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_6_union.pickle", absolute_path=True)

    # second_sample = unpickle_me_this("/Users/iason/PycharmProjects/Charm2/data_storage/fluct_is_not_constrained/real/PAPER_b0_0_2_union.pickle", absolute_path=True)

    # plot_flat_lcdm_fields(x_max=np.max(np.log(1 + z_p)), show=True, save=False)  # Creates a plot of the comparison CMB and SN fields
    # plot_charm1_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1+z_u)), show=False,
    #                                save=True)

    # For plotting purposes:
    z_p, mu_p, _ = read_data_pantheon()
    z_u, mu_u, _ = read_data_union()
    z_d, mu_d, _ = read_data_des()

    # Extract and visualize posterior samples
    posterior_realizations_list = samples

    # Signal Space Comparison Visualization
    # s_mode = calculate_approximate_mode(posterior_realizations_list, padding_operator=X, op=s)  # test function
    # post_parameters_dict = posterior_parameters(posterior_realizations_list, signal_model=s,
    #                                             upper_bound_on_fluct = 0.5)

    # visualize_posterior_histograms(post_parameters_dict, fluct)


    # xi = samples.sample_stat()[0]["xi"]
    # tmp = X(x.field()).domain
    # HT = ift.HartleyOperator(domain=tmp)
    # plt.plot(HT.adjoint(xi).val/(tmp[0]._dvol * np.sqrt(tmp[0].shape[0])))
    # plt.show()


    if plot_in_signal_space:
        if cosmological:
            if extend:
                posterior_samples = list(posterior_realizations_list.iterator(s))  # Nifty8 Field instances
                posterior_samples_cleaned = [field.val for field in posterior_samples]  # Extracted values
                s_mean, s_var = posterior_realizations_list.sample_stat(s)
                s_err = np.sqrt(s_var.val)
            else:

                posterior_samples = list(posterior_realizations_list.iterator(s))  # Nifty8 Field instances
                posterior_samples_cleaned = [X.adjoint(field).val for field in posterior_samples]  # Extracted values
                s_mean, s_var = posterior_realizations_list.sample_stat(s)
                s_err = np.sqrt(X.adjoint(s_var).val)
                s_mean = X.adjoint(s_mean)


                # s_mean_2, s_var_2 = second_sample.sample_stat(s)
                # s_mean_2 =  X.adjoint(s_mean_2)

                # latent_posterior = posterior_realizations_list.sample_stat()[0].val
                # posterior_line_slope = 2 + 5 * latent_posterior["line model slope"]
                # posterior_line_offset = 30 + 10 * latent_posterior["line model y-intercept"]
                #
                # posterior_line = linear(x.field().val, posterior_line_slope, posterior_line_offset)
                # posterior_cfm = s_mean.val - posterior_line

                # from scipy.optimize import curve_fit

                # popt = curve_fit(linear, x.field().val, s_mean.val, sigma=s_err)
                #
                # bf_line_to_signal = linear(x.field().val, *popt[0])
                # plt.title("residuals between resulting field and soley a linear model")
                # plt.plot(x.field().val, bf_line_to_signal- s_mean.val)  # max abs(res) = 0.15
                # plt.show()
                # stop
                # popt = curve_fit(flat_lcdm, x.field().val, s_mean.val, sigma=s_err)
                # popt2 = curve_fit(flat_evolving_dark_energy, x.field().val, s_mean.val, sigma=s_err)[0]

            plot_charm2_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1 + z_u)),
                                             x_max_des=np.max(np.log(1 + z_d)), show=True, save=False, x=x.field().val,
                                             s=s_mean.val, s_err=s_err, dataset_used=data_to_analyze,
                                             neg_a_mag=neg_a_mag, b0=fluct, apply_common_labels=True, disable_hist=True,
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
            plt.clf()
        else:
            raise ValueError("Synthetic comparison in signal space not implemented yet")


    if plot_in_data_space:

        posterior_samples = list(samples.iterator(s))  # Nifty8 Field instances
        posterior_samples_cleaned = [field.val for field in posterior_samples]  # Extracted values
        s_mean, s_var = posterior_realizations_list.sample_stat(s)
        s_err = np.sqrt(s_var.val)

        x_ext = x
        s_mean = X.adjoint(s_mean)
        s_var = X.adjoint(s_var)

        # Reconstruct data from posterior
        reconstructed_data = build_response(x, ift.Field.from_raw(x, s_mean.val), data_space, neg_a_mag)
        s_cmb, s_cmb_err = build_flat_lcdm(x.field().val, mode='CMB')
        s_sn, s_sn_err = build_flat_lcdm(x.field().val, mode='SN')
        data_from_cmb_lcdm = build_response(x, ift.Field.from_raw(x, s_cmb), data_space, neg_a_mag)
        data_from_sn_lcdm = build_response(x, ift.Field.from_raw(x, s_sn), data_space, neg_a_mag)

        # print("Covariance: ", cov)

        print("Chi^2_dof, base lcdm CMB: ", chi_square_dof(d.val, data_from_cmb_lcdm.val, np.linalg.inv(cov)))
        print("Chi^2_dof, base lcdm SN: ", chi_square_dof(d.val, data_from_sn_lcdm.val, np.linalg.inv(cov)))
        print("Chi^2_dof, RECONSTRUCTION: ", chi_square_dof(d.val, reconstructed_data.val, np.linalg.inv(cov)))

        plt.figure()
        # Data Space Comparison Visualization
        plt.plot(neg_a_mag, d.val, ".", color="black", label=r"Real $\mu$")
        # plt.plot(neg_a_mag, data_from_cmb_lcdm.val, ".", label="Data from base CMB", color="red")
        # plt.plot(neg_a_mag, data_from_sn_lcdm.val, ".", label="Data from base SN", color="green")
        plt.plot(neg_a_mag, reconstructed_data.val, color=blue, label="Data from reconstruction")
        plt.xlabel("$x$", fontsize=30)
        plt.ylabel(r"Distance modulus $\mu$", fontsize=30)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        pass
        # x_ext = x_ext old
        # x_ext = X @ x  # I think?
        # s_mean = s_mean
        # s_var = s_var
