# Run from root directory as: `python3.12 -m scripts.analyze_stored_data_create_central_results_2`
from scipy.constants import G
import matplotlib.gridspec as gridspec
from datetime import datetime
import os
from nifty.cl.minimization.sample_list import SampleList
from charm2 import *
import nifty.cl as ift
import re

# cosmological = True
# extend = False
# data_to_analyze = "Union2.1"
# plot_in_signal_space = True
# plot_in_data_space = False
# if cosmological:
#     from .CONFIG_cosmological import *
#     likelihood, d, neg_a_mag, arguments, x, X, s, init_pos, cov = cosmological_likelihood(data_to_use=data_to_analyze)
#     data_space = d.domain
# else:
# from .CONFIG_synthetic import *
# raise ValueError("not implemented yet")

# fluct_range = np.arange(0.1, 1, 0.1)
# fluct_range = np.linspace(.05, 0.3, 5)  # for DESY5 low fluctuations experiment
# fluct_range = [0.05]


def linear(x, m, t):
    return m * x + t


def get_mean_and_sqrt(pickled_samples, s_model, X_operator):
    s_mean, s_var = pickled_samples.sample_stat(s_model)
    s_err = np.sqrt(X_operator.adjoint(s_var).val.asnumpy())
    s_mean = X_operator.adjoint(s_mean)
    return s_mean, s_err


def run_plot(ax, data_to_analyze, samples, b0, total_figure_height=8, **kwargs):
    # get likelihood & s_model
    LH = cosmological_likelihood(data_to_use=data_to_analyze, mode="non-parametric", init_fluctuations_parameter=b0)
    properties_old_versions = (
    LH.like, LH.meta.d, LH.meta.neg_a_mag, LH.meta.s_mdl_meta, LH.meta.x, LH.meta.ZP, LH.meta.s_model, LH.meta.init_pos,
    LH.meta.noise_cov, LH.meta.dataset_name)

    likelihood_energy, d, neg_a_mag, arguments, x, X, s, initial_pos, covariance, data_to_use = properties_old_versions

    data_space = d.domain

    # compute mean + error using s_model
    s_mean, s_err = get_mean_and_sqrt(samples, s, X)

    plot_charm2_in_comparison_fields(
        x_max_pn=np.max(np.log(1 + z_p)),
        x_max_union=np.max(np.log(1 + z_u)),
        x_max_des=np.max(np.log(1 + z_d)),
        show=False, save=False,
        x=x.field().val.asnumpy(),
        s=s_mean.val.asnumpy(),
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


z_p, mu_p, _ = read_data('Pantheon+')
z_u, mu_u, _ = read_data('Union2.1')
z_d, mu_d, _ = read_data('DESY5')

# Assumes inference data are stored in folder `inferences` in the same path as the `charm2` package and that data exists
# Otherwise, please run `cosmo_inference.py` for different b0 initial parameters
directory = Path(Path(__file__).parent.parent.parent, 'inferences/')


PATHS_TO_SAMPLES = dict(
    union=[
        Path(directory, "0.2/Union2.1_non-parametric/"),
        Path(directory, "0.6/Union2.1_non-parametric/")
    ],
    pantheon=[
        Path(directory, "0.2/Pantheon+_non-parametric/"),
        Path(directory, "0.6/Pantheon+_non-parametric/")
    ],
    desy5_vincenzi=[
        Path(directory, "0.05/DESY5_non-parametric/"),
        Path(directory, "0.2/DESY5_non-parametric/"),
        Path(directory, "0.6/DESY5_non-parametric/"),
    ],
    desy5_dovekie=[
        Path(directory, "0.05/DESY5_dovekie_non-parametric/"),
        Path(directory, "0.2/DESY5_dovekie_non-parametric/"),
        Path(directory, "0.6/DESY5_dovekie_non-parametric/"),
    ]
)


def _parse_folder_name(folder_path: str):
    folder = Path(folder_path)
    folder_parts = folder.parts

    # Extract b0: assume it's the first numeric part in the path
    b0 = None
    for part in folder_parts:
        try:
            val = float(part)
            b0 = val
            break
        except ValueError:
            continue
    if b0 is None:
        raise ValueError(f"Could not find b0 in path {folder_path}")

    # Extract dataset name: must be one of the known datasets
    known_datasets = ["Union2.1", "Pantheon+", "DESY5", "DESY5_dovekie"]
    dataset_name = None
    for part in folder_parts:
        for ds in known_datasets:
            if ds in part:
                dataset_name = ds
                break
        if dataset_name is not None:
            break
    if dataset_name is None:
        raise ValueError(f"Could not find dataset name in path {folder_path}")

    return dataset_name, b0


def get_samples(folder_name):
    dataset_name, b0 = _parse_folder_name(folder_name)
    LH = cosmological_likelihood(dataset_name, mode="non-parametric", init_fluctuations_parameter=b0)
    inference_args = dict(likelihood_energy=LH.like,
                          total_iterations=1,
                          n_samples=1,
                          kl_minimizer=descent_finder,
                          sampling_iteration_controller=ic_sampling_lin,
                          nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                          return_final_position=False,
                          resume=True,
                          initial_position=None,
                          plot_energy_history=True
                          )
    posterior_samples = ift.optimize_kl(output_directory=folder_name, **inference_args)
    return LH, posterior_samples


gs = lambda x: get_samples(x)[1]

samples_union_b0_0_2, samples_union_b0_0_6 = (gs(p) for p in PATHS_TO_SAMPLES['union'])
samples_pantheon_b0_0_2, samples_pantheon_b0_0_6 = (gs(p) for p in PATHS_TO_SAMPLES['pantheon'])
samples_desy5_b0_0_05, samples_desy5_b0_0_2, samples_desy5_b0_0_6 = (gs(p) for p in PATHS_TO_SAMPLES['desy5_vincenzi'])
samples_desy5_DOV_b0_0_05, samples_desy5_DOV_b0_0_2, samples_desy5_DOV_b0_0_6 = (gs(p) for p in PATHS_TO_SAMPLES['desy5_dovekie'])

neg_a_mag_union = cosmological_likelihood("Union2.1", mode="non-parametric", init_fluctuations_parameter=0.2).meta.neg_a_mag
neg_a_mag_pantheon = cosmological_likelihood("Pantheon+", mode="non-parametric", init_fluctuations_parameter=0.2).meta.neg_a_mag
neg_a_mag_desy5 = cosmological_likelihood("DESY5", mode="non-parametric", init_fluctuations_parameter=0.2).meta.neg_a_mag

neg_a_mags_list = [neg_a_mag_union, neg_a_mag_desy5, neg_a_mag_pantheon]
x_max=1.2

# OLD; don't delete for legacy reasons
# samples_union_b0_0_2 = unpickle_me_this("/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_2_union.pickle", absolute_path=True)
# samples_union_b0_0_6 = unpickle_me_this("/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_6_union.pickle", absolute_path=True)
# samples_pantheon_b0_0_6 = unpickle_me_this(
#     "/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_6_pantheon.pickle", absolute_path=True)
# samples_pantheon_b0_0_2 = unpickle_me_this(
#     "/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_2_pantheon.pickle", absolute_path=True)
# samples_desy5_b0_0_2 = unpickle_me_this("/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_2_desy5.pickle", absolute_path=True)
# samples_desy5_b0_0_6 = unpickle_me_this("/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_6_desy5.pickle", absolute_path=True)
# samples_desy5_b0_0_05 = unpickle_me_this("/data_storage/PAPER_fluct_is_not_constrained/real/PAPER_b0_0_05_desy5.pickle", absolute_path=True)

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
now = datetime.now()
plt.savefig(f"PAPER_final_reconstructions_{now}.pdf", format="pdf", bbox_inches='tight')
# plt.show()

