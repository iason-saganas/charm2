from data_storage.style_components.matplotlib_style import *
# Run from root directory as: `python3.12 -m scripts.analyze_stored_data_create_central_results`
from scipy.constants import G, c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    # "des.pickle"  # for fluct_init = 0.6
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


def get_mean_and_sqrt(pickled_samples, s_model, X_operator):
    s_mean, s_var = pickled_samples.sample_stat(s_model)
    s_err = np.sqrt(X_operator.adjoint(s_var).val)
    s_mean = X_operator.adjoint(s_mean)
    return s_mean, s_err


z_p, mu_p, _ = read_data_pantheon()
z_u, mu_u, _ = read_data_union()
z_d, mu_d, _ = read_data_des()


def run_plot(ax, data_to_analyze, samples, b0):
    # get likelihood & s_model
    likelihood, d, neg_a_mag, arguments, x, X, s, init_pos, cov = cosmological_likelihood(data_to_use=data_to_analyze)
    data_space = d.domain

    # compute mean + error using s_model
    s_mean, s_err = get_mean_and_sqrt(samples, s, X)

    plot_charm2_in_comparison_fields(
        x_max_pn=np.max(np.log(1 + z_p)),
        x_max_union=np.max(np.log(1 + z_u)),
        x_max_des=np.max(np.log(1 + z_d)),
        show=False, save=False,
        x=x.field().val,
        s=s_mean.val,
        s_err=s_err,
        dataset_used=data_to_analyze,
        neg_a_mag=neg_a_mag,
        b0=b0,
        apply_common_labels=False,
        disable_hist=True,
        disable_x_label=True,
        disable_y_label=True,
        plot_evolving_dark_energy=False,
        custom_axs=ax
    )


def plot_hist(axis, neg_a_mag, x_max):
    hist_data = axis.hist(neg_a_mag, histtype="step", color="black", lw=0.5, bins=10)
    axis.set_ylabel(r"$\#$ of dtps", fontsize=30)
    axis.set_xlim(0, x_max)


samples_union_b0_0_6 = unpickle_me_this("/Users/iason/PycharmProjects/Charm2/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_6_union.pickle", absolute_path=True)
samples_union_b0_0_2 = unpickle_me_this("/Users/iason/PycharmProjects/Charm2/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_2_union.pickle", absolute_path=True)

samples_pantheon_b0_0_6 = unpickle_me_this("/Users/iason/PycharmProjects/Charm2/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_6_pantheon.pickle", absolute_path=True)
samples_pantheon_b0_0_2 = unpickle_me_this("/Users/iason/PycharmProjects/Charm2/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_2_pantheon.pickle", absolute_path=True)

samples_desy5_b0_0_2 = unpickle_me_this("/Users/iason/PycharmProjects/Charm2/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_2_desy5.pickle", absolute_path=True)
samples_desy5_b0_0_6 = unpickle_me_this("/Users/iason/PycharmProjects/Charm2/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_6_desy5.pickle", absolute_path=True)
samples_desy5_b0_0_05 = unpickle_me_this("/Users/iason/PycharmProjects/Charm2/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_05_desy5.pickle", absolute_path=True)

plots = [
    ("Union2.1", samples_union_b0_0_6, 0.6),
    ("Union2.1", samples_union_b0_0_2, 0.2),
    ("Pantheon+", samples_pantheon_b0_0_6, 0.6),
    ("Pantheon+", samples_pantheon_b0_0_2, 0.2),
    ("DESY5", samples_desy5_b0_0_6, 0.6),
    ("DESY5", samples_desy5_b0_0_2, 0.2),
    ("DESY5", samples_desy5_b0_0_05, 0.05),
]

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

n_cols = 3
n_data_rows = 1 + (len(plots)-1)//n_cols + 1  # adjust rows as needed
height_ratios = [0.5] + [1]*(n_data_rows-1)

fig = plt.figure(figsize=(10*n_cols, 5*n_data_rows), dpi=70)
gs = gridspec.GridSpec(n_data_rows, n_cols, figure=fig, wspace=0.1, hspace=0.1, height_ratios=height_ratios)

# Let them share the same axis
axs = []
for i in range(n_cols):
    for j in range(n_cols):
        if i == 0 and j == 0:
            ax = fig.add_subplot(gs[i,j])  # first subplot
        else:
            ax = fig.add_subplot(gs[i,j], sharex=axs[0])  # share x with first
        axs.append(ax)

# only show x labels on the bottom:

for ax in axs[:-n_cols]:  # all except bottom row
    ax.label_outer()  # hides x labels and tick labels where not needed

axs[0].set_xlim(0, 1.2)
axs[0].set_ylim(29.5, 33)

_, _, neg_a_mag_union, _, _, _, _, _, _ = cosmological_likelihood(data_to_use="Union2.1")
_, _, neg_a_mag_pantheon, _, _, _, _, _, _ = cosmological_likelihood(data_to_use="Pantheon+")
_, _, neg_a_mag_desy5, _, _, _, _, _, _ = cosmological_likelihood(data_to_use="DESY5")

# Example data for your histograms
neg_a_mags_list = [neg_a_mag_union, neg_a_mag_pantheon, neg_a_mag_desy5]  # define these
x_max = 1.2

# Fill first row with histograms
for j in range(n_cols):
    ax = axs[j]  # first row
    if j < len(neg_a_mags_list):
        plot_hist(ax, neg_a_mags_list[j], x_max)
    else:
        ax.set_visible(False)  # hide any extra axes if n>len(neg_a_mags_list)


# Now plot, skipping the first row
for i, (dataset, samples, b0) in enumerate(plots):
    row, col = divmod(i, n_cols)
    row += 1  # skip first row
    ax = fig.add_subplot(gs[row, col])
    run_plot(ax, dataset, samples, b0)

# plt.tight_layout()
plt.savefig("mytest.pdf", format="pdf", bbox_inches='tight')
plt.show()

stop

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


