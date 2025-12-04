# %% Imports and plot formatting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
Cmin, Cmax = 1e6, -1e6
Cquant = "Bq"
Sillero2014 = db.load_case(f"{fpath.Sillero2014_path}/Re_theta5000.dill")
Zhang2018 = {}
for c in glob(f"{fpath.Zhang2018_path}/*.dill"):
    cname = c.split('/')[-1][:-5]
    Zhang2018[cname] = db.load_case(c)
    Cq = getattr(Zhang2018[cname],Cquant)
    Cmin = Cq if Cq<Cmin else Cmin
    Cmax = Cq if Cq>Cmax else Cmax
    _ = Zhang2018[cname].vel_transform(label="VD")
    _ = Zhang2018[cname].vel_transform(label="TL")
    _ = Zhang2018[cname].vel_transform(label="V")
    _ = Zhang2018[cname].vel_transform(label="GFM")
Volpiani2020 = {}; Vxs = 900
for c in glob(f"{fpath.Volpiani2020_path}/*.dill"):
    cname = c.split('/')[-1][:-5]
    Volpiani2020[cname] = db.load_case(c)
    Cq = getattr(Volpiani2020[cname],Cquant)[Vxs]
    Cmin = Cq if Cq<Cmin else Cmin
    Cmax = Cq if Cq>Cmax else Cmax
    _ = Volpiani2020[cname].vel_transform(label="VD")
    _ = Volpiani2020[cname].vel_transform(label="TL")
    _ = Volpiani2020[cname].vel_transform(label="V")
    _ = Volpiani2020[cname].vel_transform(label="GFM")

# %% Plotting
col = lambda q: cm.rainbow((q-Cmin)/(Cmax-Cmin))
cmap = cm.ScalarMappable(matplotlib.colors.Normalize(vmin=Cmin, vmax=Cmax),"rainbow")

# No compressible scaling
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Zhang2018.keys():
    ax.semilogx(Zhang2018[c].yplus, Zhang2018[c].uplus, 
                color=col(getattr(Zhang2018[c],Cquant)), label=f"Z18:{c}")
for c in Volpiani2020.keys():
    ax.semilogx(Volpiani2020[c].yplus[Vxs], Volpiani2020[c].uplus[Vxs], '--',
                color=col(getattr(Volpiani2020[c],Cquant)[Vxs]), label=f"V20:{c}")
ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
            label="Sillero et al. (2014)")
# ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^+$")
ax.set_ylabel(r"$\overline{u}^+$")
ax.set_title("No Compressibility Correction")
cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(Cquant)
fig.tight_layout(); fig.show()

# Van Driest
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Zhang2018.keys():
    ax.semilogx(Zhang2018[c].yplusVD, Zhang2018[c].uplusVD, 
                color=col(getattr(Zhang2018[c],Cquant)), label=f"Z18:{c}")
for c in Volpiani2020.keys():
    ax.semilogx(Volpiani2020[c].yplusVD[Vxs], Volpiani2020[c].uplusVD[Vxs], '--',
                color=col(getattr(Volpiani2020[c],Cquant)[Vxs]), label=f"V20:{c}")
ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
            label="Sillero et al. (2014)")
# ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^+$")
ax.set_ylabel(r"$\overline{u}^+_{\mathrm{VD}}$")
ax.set_title("Van Driest Correction")
cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(Cquant)
fig.tight_layout(); fig.show()

# Trettel & Larsson
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Zhang2018.keys():
    ax.semilogx(Zhang2018[c].yplusTL, Zhang2018[c].uplusTL, 
                color=col(getattr(Zhang2018[c],Cquant)), label=f"Z18:{c}")
for c in Volpiani2020.keys():
    ax.semilogx(Volpiani2020[c].yplusTL[Vxs], Volpiani2020[c].uplusTL[Vxs], '--',
                color=col(getattr(Volpiani2020[c],Cquant)[Vxs]), label=f"V20:{c}")
ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
            label="Sillero et al. (2014)")
# ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^*$")
ax.set_ylabel(r"$\overline{u}^+_{\mathrm{TL}}$")
ax.set_title(r"Trettel \& Larsson Correction")
cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(Cquant)
fig.tight_layout(); fig.show()

# Volpiani et al.
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Zhang2018.keys():
    ax.semilogx(Zhang2018[c].yplusV, Zhang2018[c].uplusV, 
                color=col(getattr(Zhang2018[c],Cquant)), label=f"Z18:{c}")
for c in Volpiani2020.keys():
    ax.semilogx(Volpiani2020[c].yplusV[Vxs], Volpiani2020[c].uplusV[Vxs], '--', 
                color=col(getattr(Volpiani2020[c],Cquant)[Vxs]), label=f"V20:{c}")
ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
            label="Sillero et al. (2014)")
# ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^+_\mathrm{V}$")
ax.set_ylabel(r"$\overline{u}^+_{\mathrm{V}}$")
ax.set_title("Volpiani Correction")
cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(Cquant)
fig.tight_layout(); fig.show()

# Griffin et al.
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Zhang2018.keys():
    ax.semilogx(Zhang2018[c].yplusGFM, Zhang2018[c].uplusGFM, 
                color=col(getattr(Zhang2018[c],Cquant)), label=f"Z18:{c}")
for c in Volpiani2020.keys():
    ax.semilogx(Volpiani2020[c].yplusGFM[Vxs], Volpiani2020[c].uplusGFM[Vxs], '--', 
                color=col(getattr(Volpiani2020[c],Cquant)[Vxs]),label=f"V20:{c}")
ax.semilogx(Sillero2014.yplus, Sillero2014.uplus, color='black', 
            label="Sillero et al. (2014)")
# ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^*$")
ax.set_ylabel(r"$\overline{u}^+_{\mathrm{GFM}}$")
ax.set_title(r"Griffin, Fu, \& Moin Correction")
cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(Cquant)
fig.tight_layout(); fig.show()

# %%
