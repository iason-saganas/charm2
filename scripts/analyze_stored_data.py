from data_storage.style_components.matplotlib_style import *

cosmological = True
data_to_analyze = "DESY5"
plot_in_signal_space = True
plot_in_data_space = False
if cosmological:
    from .CONFIG_cosmological import *
    likelihood, d, neg_a_mag, arguments, x, X, s, cov = cosmological_likelihood(data_to_use=data_to_analyze)
    data_space = d.domain
else:
    from .CONFIG_synthetic import *
    raise ValueError("not implemented yet")


# Run from root directory as: `python3.12 -m scripts.analyze_stored_data`

samples = unpickle_me_this("real/2024-09-07_20-00-12.pickle", absolute_path=False)

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
s_mean, s_var = posterior_realizations_list.sample_stat(s)
posterior_samples = list(posterior_realizations_list.iterator(s))  # Nifty8 Field instances
posterior_samples_cleaned = [X.adjoint(field).val for field in posterior_samples]  # Extracted values
# s_mode = calculate_approximate_mode(posterior_realizations_list, padding_operator=X, op=s)  # test function

if plot_in_signal_space:
    if cosmological:
        s_err = np.sqrt(X.adjoint(s_var).val)
        plot_charm2_in_comparison_fields(x_max_pn=np.max(np.log(1 + z_p)), x_max_union=np.max(np.log(1 + z_u)),
                                         x_max_des=np.max(np.log(1 + z_d)), show=True, save=False, x=x.field().val,
                                         s=X.adjoint(s_mean).val, s_err=s_err, dataset_used=data_to_analyze,
                                         neg_a_mag=neg_a_mag, additional_samples=None)
        plt.clf()
    else:
        raise ValueError("Synthetic comparison in signal space not implemented yet")


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

    print("Covariance: ", cov)

    print("Chi^2_dof, base lcdm CMB: ", chi_square_dof(d.val, data_from_cmb_lcdm.val, np.linalg.inv(cov)))
    print("Chi^2_dof, base lcdm SN: ", chi_square_dof(d.val, data_from_sn_lcdm.val, np.linalg.inv(cov)))
    print("Chi^2_dof, RECONSTRUCTION: ", chi_square_dof(d.val, reconstructed_data.val, np.linalg.inv(cov)))

    plt.figure(figsize=(8,8))
    # Data Space Comparison Visualization
    plt.plot(neg_a_mag, d.val, ".", color="black", label=r"Real $\mu$")
    # plt.plot(neg_a_mag, data_from_cmb_lcdm.val, ".", label="Data from base CMB", color="red")
    # plt.plot(neg_a_mag, data_from_sn_lcdm.val, ".", label="Data from base SN", color="green")
    plt.plot(neg_a_mag, reconstructed_data.val, color=blue, label="Data from reconstruction")
    plt.xlabel("$x$")
    plt.ylabel(r"Distance modulus $\mu$")
    plt.legend()
    plt.show()
else:
    # x_ext = x_ext old
    x_ext = X @ x  # I think?
    s_mean = s_mean
    s_var = s_var
