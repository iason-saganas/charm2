from data_storage.style_components.matplotlib_style import *
from .CONFIG_synthetic import *
# Run from root directory as: `python3.12 -m scripts.analyze_stored_data_synthetic`

full_signal_space = False
plot_in_signal_space = True
plot_in_data_space = False

# fluct_range = np.arange(0.1, 1, 0.1)  # large fluct range
# fluct_range = np.arange(0.1, 0.5, 0.05)  # small fluct range

fluct_range = [0.6]

corresponding_pickles = ["2025-05-22_16-02-20.pickle",
"2025-05-22_16-20-11.pickle",
"2025-05-22_16-37-38.pickle",
"2025-05-22_16-55-10.pickle",
"2025-05-22_17-12-27.pickle",
"2025-05-22_17-31-49.pickle",
"2025-05-22_17-49-37.pickle",
"2025-05-22_18-07-35.pickle"]


for idx, fluct in enumerate(fluct_range):
    # pickl = corresponding_pickles[idx]
    # pickled_data = unpickle_me_this(f"/Users/iason/PycharmProjects/Charm2/data_storage/pickled_inferences/synthetic/PAPER_synthetic_bump_0_1_bump.pickle", absolute_path=True)
    pickled_data = unpickle_me_this("/Users/iason/PycharmProjects/Charm2/data_storage/PAPER_fluct_is_not_constrained/synthetic/PAPER_b0_0_6_exponential_data.pickle", absolute_path=True)

    samples, data, ground_truth_field, signal_model, neg_a_mag = pickled_data
    s=signal_model

    x_length = np.max(neg_a_mag) # ADJUST
    x_fac = 2 # ADJUST
    pxl_size = x_length / n_pix

    x = ift.RGSpace(n_pix, distances=pxl_size)  # The Signal space.
    x_ext = ift.RGSpace(x_fac*n_pix, distances=pxl_size)  # The extended signal space
    x = attach_custom_field_method(x)  # Attach `field()` method
    x_ext = attach_custom_field_method(x_ext)  # Attach `field()` method

    X = ift.FieldZeroPadder(domain=x, new_shape=(x_fac*n_pix, ))


    # plot_flat_lcdm_fields(x_max=np.max(np.log(1 + z_p)), show=True, save=False)  # Creates a plot of the comparison CMB and SN fields
    # plot_charm1_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1+z_u)), show=False,
    #                                save=True)

    # For plotting purposes:
    z_p, mu_p, _ = read_data_pantheon()
    z_u, mu_u, _ = read_data_union()
    z_d, mu_d, _ = read_data_des()

    # Extract and visualize posterior samples
    posterior_realizations_list = samples

    s_mean, s_var = posterior_realizations_list.sample_stat(s)

    posterior_samples = list(posterior_realizations_list.iterator(s))  # Nifty8 Field instances
    posterior_samples_cleaned = [X.adjoint(field).val for field in posterior_samples]  # Extracted values

    # Uncommented because CDF needs to be applied instead of exp and CDF uniform prior needs to be plotted
    # instead of lognormal prior on fluct

    # xi_s_posterior_harmonic = posterior_realizations_list.sample_stat()[0]["xi"]
    # HT = ift.HartleyOperator(domain=x_ext)
    # N = len(x_ext.field().val)
    # xi_s_posterior_real = (HT.adjoint(xi_s_posterior_harmonic)).val / (np.sqrt(N) * x_ext.distances[0])
    # plt.plot(xi_s_posterior_harmonic.val)
    # plt.loglog()
    # plt.show()

    # post_parameters_dict = posterior_parameters(posterior_realizations_list, signal_model=s, upper_bound_on_fluct=1)
    # visualize_posterior_histograms(post_parameters_dict, fluct)


def linear(x, m, t):
    return m * x + t

# Signal Space Comparison Visualization
if plot_in_signal_space:
    # Fit the line in the adjoint space where there is actual data
    reconstruct_mean = X.adjoint(s_mean).val
    reconstruct_var = X.adjoint(s_var).val
    if not full_signal_space:


        s_err = np.sqrt(X.adjoint(s_var).val)

        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[2,2,6])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        mean_prior_field = s(construct_initial_position(n_pix_ext=s_mean.domain[0].shape[0],
                                                      distances=s_mean.domain[0].distances[0],
                                                      fluctuations=0.2,
                                                      apply_prior_line_slope=True,
                                                      apply_prior_line_offset=True,
                                                      apply_prior_xi_s=True
                                                      ))

        plot_synthetic_data(neg_scale_fac_mag=neg_a_mag, data=data.val, x_max_pn=np.max(np.log(1 + z_p)),
                            mu_array=np.concatenate((mu_p, mu_u)), save=False, show=False, custom_axs = [ax1, ax2])


        plot_synthetic_ground_truth(x=x, ground_truth=X.adjoint(ground_truth_field).val, x_max_pn=np.max(np.log(1 + z_p)),
                                    reconstruction=(reconstruct_mean, np.sqrt(reconstruct_var)), save=False,
                                    show=True, custom_ax=ax3, # further_samples=[X.adjoint(mean_prior_field)],
                                    labels_further_samples=["Prior mean field"])

        stop


    else:
        reconstruct_mean = s_mean.val
        reconstruct_var = s_var.val

        latent_posterior = posterior_realizations_list.sample_stat()[0].val
        posterior_line_slope =  2 + 5*latent_posterior["line model slope"]
        posterior_line_offset =  30 + 10*latent_posterior["line model y-intercept"]

        posterior_line = linear(x_ext.field().val, posterior_line_slope, posterior_line_offset)
        posterior_cfm = reconstruct_mean - posterior_line

        # print("post cfm: ", posterior_cfm)

        # plt.plot(x_ext.field().val, posterior_cfm, color = "black", ls = '-', lw=5,
        #          label = 'Posterior CFM', alpha=0.6, markersize=0)
        #
        # plt.plot(x_ext.field().val, posterior_line, color = "black", ls = '--', lw=2, alpha=0.6, markersize=0,
        #          label="Posterior line")

        s_err = np.sqrt(s_var.val)
        plot_synthetic_ground_truth(x=x_ext, ground_truth=ground_truth_field.val,
                                    x_max_pn=np.max(np.log(1 + z_p)),
                                    reconstruction=(reconstruct_mean, np.sqrt(reconstruct_var)), save=False,
                                    show=True)
    plt.clf()

if plot_in_data_space:
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
    #
    # print("Chi^2_dof, base lcdm CMB: ", chi_square_dof(d.val, data_from_cmb_lcdm.val, np.linalg.inv(cov)))
    # print("Chi^2_dof, base lcdm SN: ", chi_square_dof(d.val, data_from_sn_lcdm.val, np.linalg.inv(cov)))
    # print("Chi^2_dof, RECONSTRUCTION: ", chi_square_dof(d.val, reconstructed_data.val, np.linalg.inv(cov)))

    plt.figure(figsize=(8,8))
    # Data Space Comparison Visualization
    plt.plot(neg_a_mag, data.val, ".", color="black", label=r"Real $\mu$")
    # plt.plot(neg_a_mag, data_from_cmb_lcdm.val, ".", label="Data from base CMB", color="red")
    # plt.plot(neg_a_mag, data_from_sn_lcdm.val, ".", label="Data from base SN", color="green")
    plt.plot(neg_a_mag, reconstructed_data.val, color=blue, label="Data from reconstruction")
    plt.xlabel("$x$")
    plt.ylabel(r"Distance modulus $\mu$")
    plt.legend()
    plt.show()
else:
    pass
    # # x_ext = x_ext old
    # x_ext = X @ x  # I think?
    # s_mean = s_mean
    #s_var = s_var
