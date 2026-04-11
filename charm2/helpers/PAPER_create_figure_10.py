import os

from charm2 import *
from matplotlib.gridspec import GridSpec
from pathlib import Path
import nifty.cl as ift
from datetime import datetime

seed = 42  # ! Needs to agree with the seed used in generating the data in the specific inference run. Otherwise
# domains will not agree
ift.random.push_sseq_from_seed(seed)
np.random.seed(seed)

_, _, des_cov = read_data_des()
alpha = 1.0
npa_b0 = 0.6

class B0NotInFolderName(Exception):
    pass

def _get_b0_from_folder_name(folder_path: str):
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
        raise B0NotInFolderName(f"Could not find b0 in path {folder_path}")
    return b0


def _get_lh_from_folder_metadata(folder: Path, b0=None):
    if b0 is None:
        print("FLAT LCDM MODE")
        mode = 'flat_LCDM'
    else:
        print("NPA MODE")
        mode = 'non-parametric'

    # GET GROUND TRUTH ARGS
    meta_data_file = Path(folder, "metadata.txt")
    meta_gt = {
        "mode": "UNKNOWN_MODE",
        "Ωm0": np.nan,
        "H0": np.nan,
        "w0": np.nan,
        "wa": np.nan
    }

    in_gt = False
    with open(meta_data_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Ground truth:"):
                in_gt = True
                continue
            if in_gt:
                if line == "" or line.startswith("Data generation:"):
                    break  # stop at next section
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().split()[0]
                    try:
                        meta_gt[key] = float(value)
                    except ValueError:
                        meta_gt[key] = value


    ground_truth_args = GroundTruthArgs(
        mode=meta_gt["mode"],
        H0=meta_gt["H0"],
        Ωm0=meta_gt["Ωm0"],
        w0=meta_gt["w0"],
        wa=meta_gt["wa"],
        b0=None,
        m0=None
    )

    # GET DATA GENERATION ARGS
    meta_dga = {
        "noise_covariance": alpha * des_cov,
        "n_dp": 500,
        "uniform_drawing": False,
        "use_des_like_data_distribution": False,
        "custom_data_shift": None,
        "apply_shift_where": None
    }

    in_data_gen = False
    with open(meta_data_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Data generation:"):
                in_data_gen = True
                continue
            if in_data_gen:
                if line == "" or line.startswith("Ground truth:"):
                    break
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().split()[0]  # drop comments
                    if value.lower() == "true":
                        meta_dga[key] = True
                    elif value.lower() == "false":
                        meta_dga[key] = False
                    else:
                        try:
                            meta_dga[key] = float(value)
                        except ValueError:
                            meta_dga[key] = value

    meta_dga["n_dp"] = int(meta_dga["n_dp"])

    data_args = DataArgs(**meta_dga)

    np.random.seed(seed) # Reinitialize at seed, otherwise different data will be drawn
    LH = synthetic_likelihood(init_fluctuations_parameter=b0, mode=mode,
                          ground_truth_args=ground_truth_args, data_generation_args=data_args,)

    return LH


def make_figure_name(meta_gt, meta_dga, b0):
    # Determine str1
    if meta_dga.use_des_like_data_distribution:
        str1 = "des_like_data"
    else:
        str1 = "uniform_data" if meta_dga.uniform_drawing else "exponential_data"

    # Format b0 with '.' → '_'
    str2 = str(b0).replace('.', '_')

    # Build final string
    now = datetime.now()
    # fig_name = f"PAPER_synthetic_{str1}_b0_{str2}_{now}"
    fig_name = f"PAPER_synthetic_{str1}_b0_{str2}"
    return fig_name


def get_samples(folder_name, b0=None):
    LH = _get_lh_from_folder_metadata(folder_name, b0=b0)
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


directory = Path(Path(__file__).parent.parent.parent, 'inferences/')

b0_str = str(npa_b0).replace('.', '_')

PATHS_TO_SAMPLES = dict(
    lcdm_model=Path(directory, f"mock_study_1/flat_LCDM/alpha_is_{alpha}/synthetic_des_like_drawing_ground_is_flat_EDE_while_s_model_is_flat_LCDM"),
    non_parametric_model=Path(directory, f"mock_study_1/npa_b0_is_{b0_str}/alpha_is_{alpha}/synthetic_des_like_drawing_ground_is_flat_EDE_while_s_model_is_non-parametric"),
)

sample_path_lcdm = PATHS_TO_SAMPLES['lcdm_model']
sample_path_npa = PATHS_TO_SAMPLES['non_parametric_model']

LH_lcdm, posterior_samples_lcdm = get_samples(sample_path_lcdm, b0=None)
LH_npa, posterior_samples_npa = get_samples(sample_path_npa, b0=npa_b0)

X_lcdm = LH_lcdm.meta.ZP.adjoint
X_npa = LH_npa.meta.ZP.adjoint

s_mean_lcdm, s_var_lcdm = posterior_samples_lcdm.sample_stat(LH_lcdm.meta.s_model)
s_mean_npa, s_var_npa = posterior_samples_npa.sample_stat(LH_npa.meta.s_model)

s_mean_arr_lcdm, s_var_arr_lcdm = (X_lcdm(s_mean_lcdm).val.asnumpy(), X_lcdm(s_var_lcdm).val.asnumpy())
s_mean_arr_npa, s_var_arr_npa = (X_npa(s_mean_npa).val.asnumpy(), X_npa(s_var_npa).val.asnumpy())

# Read in other data and plot
z_p, mu_p, _ = read_data("Pantheon+")
z_u, mu_u, _ = read_data("Union2.1")
z_d, mu_d, _ = read_data("DESY5")
x_max_pn = np.max(np.log(1 + z_p))

fontsize_of_legend = 25
fontsize_of_labels = 30

default_height = 2/3*8
default_widht = 10
fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True, figsize=(default_widht, 2 * default_height))
ax1:plt.axis = axs[0]
ax2:plt.axis = axs[1]

# args_lcdm = dict(x=LH_lcdm.meta.x, ground_truth=X_lcdm(LH_lcdm.meta.ground_truth_field).val.asnumpy(), x_max_pn=x_max_pn,
#              show_comparison_fields=False,
#              reconstruction=(s_mean_arr_lcdm, np.sqrt(s_var_arr_lcdm)),
#              special_legend="I",
#              fn="fn")
#
# args_npa = dict(x=LH_npa.meta.x, ground_truth=X_npa(LH_npa.meta.ground_truth_field).val.asnumpy(), x_max_pn=x_max_pn,
#              show_comparison_fields=False,
#              reconstruction=(s_mean_arr_npa, np.sqrt(s_var_arr_npa)),
#              special_legend="I",
#              fn="fn")

def how_much_outside_of_1_sigma(residuals, standard_deviation):
    where_inside = np.where((residuals < standard_deviation) & (residuals > -standard_deviation))[0]
    N = len(residuals)
    M = len(where_inside)
    print(r"Percentage inside of $1\sigma$:", len(where_inside) / N * 100, "%")


# Plot residuals instead
gt = X_lcdm(LH_lcdm.meta.ground_truth_field).val.asnumpy()

res_lcdm = s_mean_arr_lcdm-gt
how_much_outside_of_1_sigma(res_lcdm, np.sqrt(s_var_arr_lcdm))
args_lcdm = dict(x=LH_lcdm.meta.x, ground_truth=None, x_max_pn=x_max_pn,
             show_comparison_fields=False, indicate_end_of_data=False,
             reconstruction=(res_lcdm, np.sqrt(s_var_arr_lcdm)),
             special_legend=None, ylim=(-0.2, 0.2), xlim=(0, 0.8),
             custom_reconstruction_label=r'flat $\Lambda\mathrm{CDM}$ fit',
             fn="fn")

res_npa = s_mean_arr_npa-gt
how_much_outside_of_1_sigma(res_npa, np.sqrt(s_var_arr_npa))
args_npa = dict(x=LH_npa.meta.x, ground_truth=None, x_max_pn=x_max_pn,
             show_comparison_fields=False, indicate_end_of_data=False,
             reconstruction=(s_mean_arr_npa-gt, np.sqrt(s_var_arr_npa)),
             special_legend=None, ylim=(-0.2, 0.2), xlim=(0, 0.8),
             custom_reconstruction_label=r'$\texttt{charm2}$ non-parametric' + f', $b_0$={npa_b0}',
             fn="fn")

ax2_mini = ax2.inset_axes([0.72, 0.1, 0.25, 0.35])
for ax in [ax1, ax2, ax2_mini]:
    x = LH_npa.meta.x.field().val.asnumpy()
    ax.plot(x, np.zeros_like(x), "-", color="black", lw=1)

plot_synthetic_ground_truth(save=False, show=False, custom_ax=ax1, **args_lcdm)
plot_synthetic_ground_truth(save=False, show=False, custom_ax=ax2, **args_npa)
plot_synthetic_ground_truth(save=False, show=False, custom_ax=ax2_mini, reconstruction_lw=1, **args_npa)


ax1.set_ylabel(r"$\Delta s(x)$", fontsize=fontsize_of_labels)
ax2.set_ylabel(r"$\Delta s(x)$", fontsize=fontsize_of_labels)
ax2.set_xlabel(r"$x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$")
ax2_mini.tick_params(labelsize=10)

for ax in axs:
    ax.legend(fontsize=fontsize_of_legend)

plt.tight_layout()
# plt.show()

current_file = Path(__file__)
project_root = current_file.parents[2]
fig_dir = Path(project_root, "figures")
os.makedirs(fig_dir, exist_ok=True)
fn = f"PAPER_ede_gt_vs_lcdm_and_npa.pdf"
filename = str(Path(fig_dir, fn))
plt.savefig(filename)
