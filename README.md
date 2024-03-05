CHARM2.0: Cosmic History Agnostic Reconstruction Method
=================
Reconstructing cosmic expansion history (cosmic energy density as a function of redshift) 
without assuming a specific type of matter distribution in the Universe. Data are distance moduli and
redshifts from Supernovae Ia from the 
[Union2.1](https://supernova.lbl.gov/Union/) compilation and [Pantheon+](https://pantheonplussh0es.github.io) Analysis.

Powered by [NIFTy](https://ift.pages.mpcdf.de/nifty/user/installation.html) (v8). 

Original [CHARM](https://gitlab.mpcdf.mpg.de/natalia/charm) by [Natalia Porqueres et al. 2017](https://arxiv.org/abs/1608.04007) + geoVI instead of an iterative MAP approach.


Requirements
=================
*   NIFTy can be installed using pip:

        pip install nifty8
* If necessary, after NIFTy installation: Numpy, Scipy, Pandas, pickle, matplotlib, seaborn 

> **Note**
> 
> Pickle files **will** be overwritten if new inference run starts and pickles are not manually renamed.
> See folder structure down below.

> **Warning**
> 
>When visualizing the power spectrum of the posterior by using the function
> `visualize_and_analyze_posterior_power_spectrum()`, the mean and uncertainty of 
> `fluctuations` and `loglogavgslope` is returned. This **only holds** if the parameters of the correlated field model
> have no standard deviation, i.e. the second number in the parameter tuple is e.g. $10^{-16}$. These parameters are found 
> by fitting the posterior power spectrum in a log-log coordinate system and use a preliminary numerical relation that will be replaced in the 
> future, so interpret these values with caution. This numerical relation also only holds for fixed length of the signal 
> domain, which has been chosen here to be $x_{max}=6.7$.




Usage and Workflow
=================
*   Download a local copy of this project per the <span style="color:green"> green code button above </span> (e.g. as zip). Unpack, move into directory and run 

        python synthetic_catalogue.py
* There will be four matplotlib figures showing up, each need to be closed for the program to continue running
* Inference run with synthetic data: `synthetic_catalogue.py`. Inference run with real data: `cosmological_tomography.py`
* After inference run, posterior samples can be re-visualized in the program `visualize_pickle_data.py` in the 📂`pickles` subfolder, by calling

        posterior_realizations = unpickle_me_this("synthetic_posterior_realizations.pickle")
* For the visualization from the pickles directory, the keyword `from_pickles_folder` will be have to be set to `True` for each plot function
* At the beginning of the inference scripts following boolean variables can be set: `switch1-8`; set if visualizations
of data, posterior, power spectrum etc. should be shown via `plt.show()` or saved in the 📂`figures` folder.
`use_union_data`; Uses Union2.1 data if `True`, Pantheon+ data if `False`


```
.
├── 🐍 cosmological_tomography.py    # Inference with real data
├── 🐍 synthetic_catalogue.py        # Inference with synthetic data 
├── 📂 figures                       # Place where figures are stored. Contains figures examples.
├── 📂 raw data                      # Folder containing moduli, redshifts, covariances ...
├── 📂 helpers                       
│   ├── 🐍 plotters.py               # Convenience plotting functions
│   └── 🐍 functions.py              # Important custom operators and other helper functions
│   
└── 📂pickles                        # Important subfolder: Contains pickles (serialized python objects) from posterior samples.
    └── 🐍 visualize_pickle_data.py  # After inference run, posterior samples can be re-visualized here.
```


### Elements of time

>Runtime synthetic data $\approx 3-5 \hspace{1mm}$ min
> 
>Runtime Union2.1 data $\approx 40-60 \hspace{1mm}$ min
> 
>Runtime Pantheon+ data $\approx 10-15 \hspace{1mm}$ hrs



