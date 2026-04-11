from charm2 import *
from matplotlib.gridspec import GridSpec
from pathlib import Path
import nifty.cl as ift
from datetime import datetime

seed = 42  # ! Needs to agree with the seed used in generating the data in the specific inference run. Otherwise
# domains will not agree
ift.random.push_sseq_from_seed(seed)
np.random.seed(seed)


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
        raise ValueError(f"Could not find b0 in path {folder_path}")
    return b0


def _get_lh_from_folder_metadata(folder: Path):
    b0 = _get_b0_from_folder_name(folder)

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
        "noise_covariance": None,
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

    LH = synthetic_likelihood(init_fluctuations_parameter=b0, mode='non-parametric',
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


def get_samples(folder_name):
    LH = _get_lh_from_folder_metadata(folder_name)
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

PATHS_TO_SAMPLES = dict(
    figure4=Path(directory, "0.2/synthetic_uniform_drawing_ground_is_flat_LCDM_while_s_model_is_non-parametric"),
    figure5a=Path(directory, "0.2/synthetic_exponential_drawing_ground_is_flat_LCDM_while_s_model_is_non-parametric"),
    figure5b=Path(directory, "0.6/synthetic_exponential_drawing_ground_is_flat_LCDM_while_s_model_is_non-parametric"),
    figure8=Path(directory, "0.6/synthetic_des_like_drawing_ground_is_flat_LCDM_while_s_model_is_non-parametric"),
)

sample_path = PATHS_TO_SAMPLES['figure8']

LH, posterior_samples = get_samples(sample_path)

X = LH.meta.ZP.adjoint
init_field = X(LH.meta.s_model(LH.meta.init_pos))
s_mean, s_var = posterior_samples.sample_stat(LH.meta.s_model)
s_mean_arr, s_var_arr = (X(s_mean).val.asnumpy(), X(s_var).val.asnumpy())
fn = make_figure_name(meta_gt=LH.meta.ground_truth_args, meta_dga=LH.meta.data_generation_args, b0=LH.meta.b0)

# Read in other data and plot
z_p, mu_p, _ = read_data("Pantheon+")
z_u, mu_u, _ = read_data("Union2.1")
z_d, mu_d, _ = read_data("DESY5")
x_max_pn = np.max(np.log(1 + z_p))


args1 = dict(neg_scale_fac_mag=LH.meta.neg_a_mag, data=LH.meta.d.val.asnumpy(), x_max_pn=x_max_pn, mu_array=np.concatenate((mu_p, mu_u)))
args2 = dict(x=LH.meta.x, ground_truth=X(LH.meta.ground_truth_field).val.asnumpy(), x_max_pn=x_max_pn,
             # further_samples=[init_field], labels_further_samples=['Initial field'],
             show_comparison_fields=True,
             reconstruction=(s_mean_arr, np.sqrt(s_var_arr)),
             special_legend="IV",
             fn=fn)


fig = plt.figure(figsize=(10,10))
gs = GridSpec(ncols=1, nrows=3, height_ratios=[2,2,6])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

plot_synthetic_data(save=False, show=False, custom_axs = [ax1, ax2], **args1)
plot_synthetic_ground_truth(save=True, show=False, custom_ax=ax3, **args2)