from .helpers import *
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from .style_components.matplotlib_style import *
style_path = Path(__file__).parent / "style_components" / "standardStyle.mplstyle"
plt.style.use(style_path)

def pointwise_CDF(x):
    return norm.cdf(x)

def pointwise_CDF_and_derv(x):
    return norm.cdf(x), 1/np.sqrt(2*np.pi)*np.exp(-1/2*x**2)

# from nifty8.pointwise import ptw_dict

from nifty.cl.pointwise import ptw_dict
ptw_dict["CDF"] = pointwise_CDF, pointwise_CDF_and_derv

# __all__ = ['ptw_dict']