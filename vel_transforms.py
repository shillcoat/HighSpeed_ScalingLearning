# %% Imports and plot formatting
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import matplotlib
# matplotlib.use('TkAgg')  # For interactive plot
plt_template = {
    "text.usetex": True
}
matplotlib.rcParams.update(plt_template)

from database import db_config as db
import database.filepaths as fpath

# %% Load cases
Sillero2014 = db.load_case(f"{fpath.Sillero2014_path}/Re_theta5000.dill")
Zhang2018 = {}
for c in glob(f"{fpath.Zhang2018_path}/*.dill"):
    cname = c.split('/')[-1][:-5]
    Zhang2018[cname] = db.load_case(c)
    assert all(Zhang2018[cname].hasdata(['yplus','uplus'])), f"Case {cname} is missing yplus and/or uplus"

# %% Plotting
# No compressible scaling
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
            label="Sillero et al. (2014)")
for c in Zhang2018.keys():
    ax.semilogx(Zhang2018[c].yplus, Zhang2018[c].uplus, label=f"Z18: {c}")
ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^+$")
ax.set_ylabel(r"$\overline{u}^+$")
fig.tight_layout(); fig.show()

# Van Driest

# Trettel & Larsson

# Griffin et al.

# %%
