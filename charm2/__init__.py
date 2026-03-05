from .helpers import *

import matplotlib.pyplot as plt
from pathlib import Path
style_path = Path(__file__).parent / "style_components" / "standardStyle.mplstyle"
plt.style.use(style_path)