from charm2.helpers.utilitites import draw_hubble_diagrams
from pathlib import Path
import os
fn = "PAPER_histograms.pdf"
current_file = Path(__file__)

project_root = current_file.parents[2]
fig_dir = Path(project_root, "figures")
os.makedirs(fig_dir, exist_ok=True)
filename = str(Path(fig_dir, fn))

draw_hubble_diagrams(save=True, show=False, only_show_hist=True, fn=filename)