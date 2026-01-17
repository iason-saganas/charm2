from data_storage.style_components.matplotlib_style import *
# Run from root directory as: `python3.12 -m scripts.analyze_stored_data_create_central_results_2`
from scipy.constants import G, c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .utilitites import special_legend_III

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
def run_plot(ax, data_to_analyze, samples, b0, total_figure_height=8, **kwargs):
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
        disable_hist=True,
        custom_axs=ax,
        total_figure_height=total_figure_height,
        **kwargs
    )


def plot_hist(axis, neg_a_mag, x_max, set_ylabel=True, total_figure_height=8):
    if axis is None:
        reconstruction_height = total_figure_height * 2/3
        hist_height = reconstruction_height/2
        # hist_height + reconstruction_height - total_figure_height == 0 True
        fig, axis = plt.subplots(figsize=(10, hist_height))
    hist_data = axis.hist(neg_a_mag, histtype="step", color="black", lw=0.5, bins=10)
    if set_ylabel:
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
    ("DESY5", samples_desy5_b0_0_6, 0.6),
    ("Pantheon+", samples_pantheon_b0_0_6, 0.6),
    ("Union2.1", samples_union_b0_0_2, 0.2),
    ("DESY5", samples_desy5_b0_0_2, 0.2),
    ("Pantheon+", samples_pantheon_b0_0_2, 0.2),
    ("Charm1", None, None),
    ("DESY5", samples_desy5_b0_0_05, 0.05),
]

_, _, neg_a_mag_union, _, _, _, _, _, _ = cosmological_likelihood(data_to_use="Union2.1")
_, _, neg_a_mag_pantheon, _, _, _, _, _, _ = cosmological_likelihood(data_to_use="Pantheon+")
_, _, neg_a_mag_desy5, _, _, _, _, _, _ = cosmological_likelihood(data_to_use="DESY5")


neg_a_mags_list = [neg_a_mag_union, neg_a_mag_desy5, neg_a_mag_pantheon]
x_max=1.2

"""
Example plots: Singular histograms and reconstruction plots
"""
# for neg_a_mag in neg_a_mags_list:
#     ax=None
#     plot_hist(ax, neg_a_mag, x_max)
#     plt.tight_layout()
#     plt.show()
#
# for i, (dataset, samples, b0) in enumerate(plots):
#     ax = None
#     run_plot(ax, dataset, samples, b0)
#     plt.tight_layout()
#     plt.show()

"""
Scaffolding for tiling said single plots
"""
n_cols = 3
n_rows = 4
total_figure_height = 8   # internal height
fontsize_of_legend = 25
fontsize_of_labels = 30
height_ratios = [0.5, 1, 1, 1]   # hist row half height

fig = plt.figure(figsize=(10*n_cols, total_figure_height*n_rows), dpi=70)  # construct one big image to be rescaled
# later
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                      height_ratios=height_ratios,
                      wspace=0.2, hspace=0.2)  # corresponding gridspec

axs = []  # collect all axes..
for i in range(n_rows):
    for j in range(n_cols):
        axs.append(fig.add_subplot(gs[i, j]))

hist_axes = axs[:3]
reconstruction_axes = axs[3:]

"""
Fill in the histograms
"""
for hist_ax, neg_a_mag in zip(hist_axes, neg_a_mags_list):
    # Only y label on the first column
    plot_hist(hist_ax, neg_a_mag, x_max, set_ylabel=False, total_figure_height=total_figure_height)


"""
Fill in the reconstruction plots
"""

special_kwargs = {
    0: {"apply_common_labels": True, "plot_evolving_dark_energy": False},
    1: {"apply_common_labels": False, "plot_evolving_dark_energy": True, "plot_de_label": True},
    2: {"apply_common_labels": False, "plot_evolving_dark_energy": False},
    3: {"apply_common_labels": False, "plot_evolving_dark_energy": False},
    4: {"apply_common_labels": False, "plot_evolving_dark_energy": True},
    5: {"apply_common_labels": False, "plot_evolving_dark_energy": False},
    6: {"apply_common_labels": False, "plot_evolving_dark_energy": False},
    7: {"apply_common_labels": False, "plot_evolving_dark_energy": True},
}
for i, (reconstruction_ax, plot_object) in enumerate(zip(reconstruction_axes, plots)):
    dataset, samples, b0 = plot_object
    reconstruction_ax.set_ylim(29.5, 33)
    if dataset == "Charm1":
        if i != 6:
            raise ValueError("Ordering error.")
        plot_charm1_in_comparison_fields(show=False, save=False, disable_hist=True, apply_common_labels=False,
                                         custom_ax=reconstruction_ax)
        reconstruction_ax.legend(fontsize=fontsize_of_legend, loc="upper left")
    else:
        kwargs_to_pass = special_kwargs[i]
        run_plot(reconstruction_ax, dataset, samples, b0, total_figure_height=total_figure_height, **kwargs_to_pass)

        # print("\n\n")
        # print("reconstruction ax labels: ", reconstruction_ax.get_legend_handles_labels())
        # print("\n\n")
        special_legend_III(plot_evolving_dark_energy=kwargs_to_pass["plot_evolving_dark_energy"],
                           custom_axs=reconstruction_ax, fontsize=fontsize_of_legend)


"""
Remove all x labels and y labels and afterwards add them where they need to be 
"""

labels_to_add = [
    {"id": (1,1), "x_label": None, "y_label": r"$\#$ of dtps"},
    {"id": (2,1), "x_label": None, "y_label": "$s(x)$"},
    {"id": (3,1), "x_label": None, "y_label": "$s(x)$"},
    {"id": (4,1), "x_label": r"$x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$", "y_label": "$s(x)$"},
    {"id": (4,2), "x_label": r"$x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$", "y_label": None},
    {"id": (3,3), "x_label": r"$x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$", "y_label": None},
]

for label_info in labels_to_add:
    row, col = label_info["id"]
    ax = axs[(row-1) * n_cols + (col-1)]   # convert 1-based to 0-based

    if label_info["x_label"] is not None:
        ax.set_xlabel(label_info["x_label"], fontsize=fontsize_of_labels)

    if label_info["y_label"] is not None:
        ax.set_ylabel(label_info["y_label"], fontsize=fontsize_of_labels)

# Destroy all unneccessary tick labels (skip histograms)
for ax in axs[3:]:
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    if xlabel == "":
        ax.set_xticklabels([])
    if ylabel == "":
        ax.set_yticklabels([])

# But also disable x tick labels on the histogram
for ax in axs[:3]:
    ax.set_xticklabels([])


"""
Extras
"""

for ax in axs[-1:]:
        ax.set_visible(False)  # hide the extra axes on the very

for ax in axs:
    ax.set_xlim(0, max(neg_a_mag_pantheon))
    # ax.legend()  # calling legend_III() above, so uncommenting will overwrite changes

fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

"""
Show plot
"""

global_fac = 2.1
width_fac = 1.25
fig.set_size_inches(12*global_fac*width_fac, 8*global_fac)   # shrink the whole thing
plt.savefig("PAPER_final_reconstructions.pdf", format="pdf", bbox_inches='tight')
# plt.show()

