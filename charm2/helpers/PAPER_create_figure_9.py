import os

from plotting_helpers_for_paper_figures import *

fontsize_of_legend = 25
fontsize_of_labels = 30

b0 = 0.6  # change here

samples_desy5_vinc_b0_0_05, samples_desy5_vinc_b0_0_2, samples_desy5_vinc_b0_0_6 = (gs(p) for p in PATHS_TO_SAMPLES['desy5_vincenzi'])
samples_desy5_dov_b0_0_05, samples_desy5_dov_b0_0_2, samples_desy5_dov_b0_0_6 = (gs(p) for p in PATHS_TO_SAMPLES['desy5_dovekie'])

if b0 == 0.6:
    samples_des = samples_desy5_vinc_b0_0_6
    samples_dov = samples_desy5_dov_b0_0_6
elif b0 == 0.2:
    samples_des = samples_desy5_vinc_b0_0_2
    samples_dov = samples_desy5_dov_b0_0_2
elif b0 == 0.05:
    samples_des = samples_desy5_vinc_b0_0_05
    samples_dov = samples_desy5_dov_b0_0_05
else:
    raise ValueError("b0 must be either 0.6 or 0.2 or 0.05")


default_height = 2/3*8
default_widht = 10
fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True, figsize=(default_widht, 2 * default_height))
ax1:plt.axis = axs[0]
ax2:plt.axis = axs[1]


run_plot(ax=ax1, data_to_analyze="DESY5", samples=samples_des, b0=b0,
         **dict(
             disable_x_label=True,
             apply_common_labels=True,
             plot_evolving_dark_energy=True,
             plot_de_label=True
         ))
run_plot(ax=ax2, data_to_analyze="DESY5_dovekie", samples=samples_dov, b0=b0,
         **dict(
             apply_common_labels=False,
             plot_evolving_dark_energy=True,
             plot_de_label=False
         ))
ax1.set_ylabel(r"$s(x)$", fontsize=fontsize_of_labels)
ax2.set_ylabel(r"$s(x)$", fontsize=fontsize_of_labels)
ax2.set_xlabel(r"$x=-\mathrm{ln}(a)=\mathrm{ln}(1+z)$")

special_legend_III(plot_evolving_dark_energy=False, custom_axs=ax1, fontsize=fontsize_of_legend)
ax2.legend(fontsize=fontsize_of_legend, loc="upper left")

# ax2.set_axis_on()
ax2.xaxis.set_visible(True)

plt.tight_layout()
# plt.show()

current_file = Path(__file__)
project_root = current_file.parents[2]
fig_dir = Path(project_root, "figures")
os.makedirs(fig_dir, exist_ok=True)
fn = f"PAPER_des_vincenci_vs_dovekie_b0_{b0}.pdf"
filename = str(Path(fig_dir, fn))
plt.savefig(filename)