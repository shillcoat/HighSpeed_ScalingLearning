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
    _ = Zhang2018[cname].vel_transform(label="VD")
    _ = Zhang2018[cname].vel_transform(label="TL")
    _ = Zhang2018[cname].vel_transform(label="V")
    # _ = Zhang2018[cname].vel_transform(label="GFM")

# %% Plotting
# No compressible scaling
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Zhang2018.keys():
    ax.semilogx(Zhang2018[c].yplus, Zhang2018[c].uplus, label=f"Z18: {c}")
ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
            label="Sillero et al. (2014)")
ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^+$")
ax.set_ylabel(r"$\overline{u}^+$")
ax.set_title("No Compressibility Correction")
fig.tight_layout(); fig.show()

# Van Driest
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Zhang2018.keys():
    ax.semilogx(Zhang2018[c].yplusVD, Zhang2018[c].uplusVD, label=f"Z18: {c}")
ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
            label="Sillero et al. (2014)")
ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^+$")
ax.set_ylabel(r"$\overline{u}^+_{\mathrm{VD}}$")
ax.set_title("Van Driest Correction")
fig.tight_layout(); fig.show()

# Trettel & Larsson
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Zhang2018.keys():
    ax.semilogx(Zhang2018[c].yplusTL, Zhang2018[c].uplusTL, label=f"Z18: {c}")
ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
            label="Sillero et al. (2014)")
ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^*$")
ax.set_ylabel(r"$\overline{u}^+_{\mathrm{TL}}$")
ax.set_title(r"Trettel \& Larsson Correction")
fig.tight_layout(); fig.show()

# Volpiani et al.
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Zhang2018.keys():
    ax.semilogx(Zhang2018[c].yplusV, Zhang2018[c].uplusV, label=f"Z18: {c}")
ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
            label="Sillero et al. (2014)")
ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^+_\mathrm{V}$")
ax.set_ylabel(r"$\overline{u}^+_{\mathrm{V}}$")
ax.set_title("Volpiani Correction")
fig.tight_layout(); fig.show()

# # Griffin et al.
# fig, ax = plt.subplots(figsize=[6,4])
# ax.grid(which='both', color='0.9')
# for c in Zhang2018.keys():
#     ax.semilogx(Zhang2018[c].yplusGFM, Zhang2018[c].uplusGFM, label=f"Z18: {c}")
# ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
#             label="Sillero et al. (2014)")
# ax.legend(loc='best', framealpha=1)
# ax.set_xlabel(r"$y^+$")
# ax.set_ylabel(r"$\overline{u}^+_{\mathrm{GFM}}$")
# ax.set_title("Griffin, Fu, & Moin Correction")
# fig.tight_layout(); fig.show()

# %%
