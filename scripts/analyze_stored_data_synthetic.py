from data_storage.style_components.matplotlib_style import *
from .CONFIG_synthetic import *
# Run from root directory as: `python3.12 -m scripts.analyze_stored_data_synthetic`
from scipy.optimize import curve_fit

full_signal_space = False
plot_in_signal_space = True
plot_in_data_space = False

pickled_data = unpickle_me_this("real/further_analysis/synth7.pickle", absolute_path=False)
# pickled_data = unpickle_me_this("synthetic/2025-04-30_11-40-56.pickle", absolute_path=False)

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
post_parameters_dict = posterior_parameters(posterior_realizations_list, signal_model=s, upper_bound_on_fluct=1)
visualize_posterior_histograms(post_parameters_dict)

def linear(x, m, t):
    return m * x + t

# Signal Space Comparison Visualization
if plot_in_signal_space:
    # Fit the line in the adjoint space where there is actual data
    reconstruct_mean = X.adjoint(s_mean).val
    reconstruct_var = X.adjoint(s_var).val
    # bf_line_param = curve_fit(linear, x.field().val, reconstruct_mean, sigma=np.sqrt(reconstruct_var))[0]
    bf_line_param = (3, 32)
    if not full_signal_space:
        bf_line = linear(x.field().val, *bf_line_param)
        approximate_posterior_mean_cfm =  reconstruct_mean - bf_line + 34
        # plt.plot(x.field().val, approximate_posterior_mean_cfm, color = "black", ls = '-', lw=1, label = 'CFM-part of model',
        #          alpha=0.6, markersize=0)

        s_err = np.sqrt(X.adjoint(s_var).val)
        plot_synthetic_ground_truth(x=x, ground_truth=X.adjoint(ground_truth_field).val, x_max_pn=np.max(np.log(1 + z_p)),
                                    reconstruction=(reconstruct_mean, np.sqrt(reconstruct_var)), save=False,
                                    show=True)
        plot_synthetic_data(neg_scale_fac_mag=neg_a_mag, data=data.val, x_max_pn=np.max(np.log(1 + z_p)),
                            mu_array=np.concatenate((mu_p, mu_u)), save=False, show=True)
    else:
        reconstruct_mean = s_mean.val
        reconstruct_var = s_var.val

        bf_line = linear(x_ext.field().val, *bf_line_param)
        approximate_posterior_mean_cfm = reconstruct_mean - bf_line + 34
        plt.plot(x_ext.field().val, approximate_posterior_mean_cfm, color = "black", ls = '-', lw=1,
                 label = 'CFM-part of model', alpha=0.6, markersize=0)
        plt.plot(x_ext.field().val, bf_line, color = "black", ls = '--', lw=1, alpha=0.6, markersize=0,
                 label="line part of the model")

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
    # x_ext = x_ext old
    x_ext = X @ x  # I think?
    s_mean = s_mean
    s_var = s_var
