from charm2 import *
import nifty.cl as ift

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


gs = lambda x: get_samples(x)[1]
x_max=1.2