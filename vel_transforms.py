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
Trettel2016 = {}
for c in glob(f"{fpath.Trettel2016_path}/*.dill"):
    cname = c.split('/')[-1][:-5]
    Trettel2016[cname] = db.load_case(c)
    Cq = getattr(Trettel2016[cname],Cquant)
    Cmin = np.min(Cq) if np.min(Cq)<Cmin else Cmin
    Cmax = np.max(Cq) if np.max(Cq)>Cmax else Cmax
    _ = Trettel2016[cname].vel_transform(label="VD")
    _ = Trettel2016[cname].vel_transform(label="TL")
    _ = Trettel2016[cname].vel_transform(label="V")
    _ = Trettel2016[cname].vel_transform(label="GFM")
Modesti2016 = {}
for c in glob(f"{fpath.Modesti2016_path}/*.dill"):
    cname = c.split('/')[-1][:-5]
    Modesti2016[cname] = db.load_case(c)
    Cq = getattr(Modesti2016[cname],Cquant)
    Cmin = np.min(Cq) if np.min(Cq)<Cmin else Cmin
    Cmax = np.max(Cq) if np.max(Cq)>Cmax else Cmax
    _ = Modesti2016[cname].vel_transform(label="VD")
    _ = Modesti2016[cname].vel_transform(label="TL")
    _ = Modesti2016[cname].vel_transform(label="V")
    _ = Modesti2016[cname].vel_transform(label="GFM")
# Zhang2018 = {}
# for c in glob(f"{fpath.Zhang2018_path}/*.dill"):
#     cname = c.split('/')[-1][:-5]
#     Zhang2018[cname] = db.load_case(c)
#     Cq = getattr(Zhang2018[cname],Cquant)
#     Cmin = np.min(Cq) if np.min(Cq)<Cmin else Cmin
#     Cmax = np.max(Cq) if np.max(Cq)>Cmax else Cmax
#     _ = Zhang2018[cname].vel_transform(label="VD")
#     _ = Zhang2018[cname].vel_transform(label="TL")
#     _ = Zhang2018[cname].vel_transform(label="V")
#     _ = Zhang2018[cname].vel_transform(label="GFM")
# Volpiani2020 = {}; Vxs = 900
# for c in glob(f"{fpath.Volpiani2020_path}/*.dill"):
#     cname = c.split('/')[-1][:-5]
#     Volpiani2020[cname] = db.load_case(c)
#     Cq = getattr(Volpiani2020[cname],Cquant)[Vxs]
#     Cmin = np.min(Cq) if np.min(Cq)<Cmin else Cmin
#     Cmax = np.max(Cq) if np.max(Cq)>Cmax else Cmax
#     _ = Volpiani2020[cname].vel_transform(label="VD")
#     _ = Volpiani2020[cname].vel_transform(label="TL")
#     _ = Volpiani2020[cname].vel_transform(label="V")
#     _ = Volpiani2020[cname].vel_transform(label="GFM")
# Wenzel2019 = {}; Retau_plt = 490
# for c in glob(f"{fpath.Wenzel2019_path}/*.dill"):
#     cname = c.split('/')[-1][:-5]
#     dat = db.load_case(c)
#     # if Retau_plt>dat.Retau[-1] or Retau_plt<dat.Retau[0]: continue  # Outside of range of Retau covered by data
#     Wenzel2019[cname] = dat
#     Cq = getattr(Wenzel2019[cname],Cquant)
#     Cmin = np.min(Cq) if np.min(Cq)<Cmin else Cmin
#     Cmax = np.max(Cq) if np.max(Cq)>Cmax else Cmax
#     _ = Wenzel2019[cname].vel_transform(label="VD")
#     _ = Wenzel2019[cname].vel_transform(label="TL")
#     _ = Wenzel2019[cname].vel_transform(label="V")
#     _ = Wenzel2019[cname].vel_transform(label="GFM")
# Nicholson2024 = {}  # Not good with current data: no meaningful Bk
# for c in glob(f"{fpath.Nicholson2019_path}/*.dill"):
#     cname = c.split('/')[-1][:-5]
#     Nicholson2024[cname] = db.load_case(c)
#     Cq = getattr(Nicholson2024[cname],Cquant)
#     Cmin = np.min(Cq) if np.min(Cq)<Cmin else Cmin
#     Cmax = np.max(Cq) if np.max(Cq)>Cmax else Cmax

#     # Cheat and copy over Favre-averaged values to use in transforms as I don't have Reynolds averaged values
#     Nicholson2024[cname].u = Nicholson2024[cname].u_F
#     Nicholson2024[cname].mu = Nicholson2024[cname].mu_F

#     _ = Nicholson2024[cname].vel_transform(label="VD")
#     _ = Nicholson2024[cname].vel_transform(label="TL")
#     _ = Nicholson2024[cname].vel_transform(label="V")
#     _ = Nicholson2024[cname].vel_transform(label="GFM")

# %% Plotting
col = lambda q: cm.rainbow((q-Cmin)/(Cmax-Cmin))
cmap = cm.ScalarMappable(matplotlib.colors.Normalize(vmin=Cmin, vmax=Cmax),"rainbow")

# No compressible scaling
fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Trettel2016.keys():
    ax.semilogx(Trettel2016[c].yplus[0], Trettel2016[c].uplus[0], 
                color=col(getattr(Trettel2016[c],Cquant)), label=f"TL16:{c}")
for c in Modesti2016.keys():
    ax.semilogx(Modesti2016[c].yplus[0], Modesti2016[c].uplus[0], 
                color=col(getattr(Modesti2016[c],Cquant)), label=f"M16:{c}")
# for c in Zhang2018.keys():
#     ax.semilogx(Zhang2018[c].yplus[0], Zhang2018[c].uplus[0], 
#                 color=col(getattr(Zhang2018[c],Cquant)), label=f"Z18:{c}")
# for c in Volpiani2020.keys():
#     ax.semilogx(Volpiani2020[c].yplus[Vxs], Volpiani2020[c].uplus[Vxs], '--',
#                 color=col(getattr(Volpiani2020[c],Cquant)[Vxs]), label=f"V20:{c}")
# for c in Wenzel2019.keys():
#     # xi = np.where(Wenzel2019[c].Retau>=Retau_plt)[0][0]
#     lstyle = '-' if Wenzel2019[c].Minf[0]>1.0 else '--'
#     for xi in range(0,len(Wenzel2019[c].x),len(Wenzel2019[c].x)//6):
#         ax.semilogx(Wenzel2019[c].yplus[xi], Wenzel2019[c].uplus[xi], lstyle,
#                     color=col(getattr(Wenzel2019[c],Cquant)), label=f"W19:{c}",
#                     alpha=1.0 if Wenzel2019[c].Minf[0]>1.0 else 0.3)
# for c in Nicholson2024.keys():
#     for xi in range(len(Nicholson2024[c].x)):
#         ax.semilogx(Nicholson2024[c].yplus[xi], Nicholson2024[c].uplus[xi], '-',
#                 color=col(getattr(Nicholson2024[c],Cquant)[xi]), label=f"N24:{c}")
ax.semilogx(Sillero2014.yplus[0], Sillero2014.uplus[0], color='black', 
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
for c in Trettel2016.keys():
    ax.semilogx(Trettel2016[c].yplusVD[0], Trettel2016[c].uplusVD[0], 
                color=col(getattr(Trettel2016[c],Cquant)), label=f"TL16:{c}")
for c in Modesti2016.keys():
    ax.semilogx(Modesti2016[c].yplusVD[0], Modesti2016[c].uplusVD[0], 
                color=col(getattr(Modesti2016[c],Cquant)), label=f"M16:{c}")
# for c in Zhang2018.keys():
#     ax.semilogx(Zhang2018[c].yplusVD[0], Zhang2018[c].uplusVD[0], 
#                 color=col(getattr(Zhang2018[c],Cquant)), label=f"Z18:{c}")
# for c in Volpiani2020.keys():
#     ax.semilogx(Volpiani2020[c].yplusVD[Vxs], Volpiani2020[c].uplusVD[Vxs], '--',
#                 color=col(getattr(Volpiani2020[c],Cquant)[Vxs]), label=f"V20:{c}")
# for c in Wenzel2019.keys():
#     # xi = np.where(Wenzel2019[c].Retau>=Retau_plt)[0][0]
#     lstyle = '-' if Wenzel2019[c].Minf[0]>1.0 else '--'
#     for xi in range(0,len(Wenzel2019[c].x),len(Wenzel2019[c].x)//6):
#         ax.semilogx(Wenzel2019[c].yplusVD[xi], Wenzel2019[c].uplusVD[xi], lstyle,
#                     color=col(getattr(Wenzel2019[c],Cquant)), label=f"W19:{c}",
#                     alpha=1.0 if Wenzel2019[c].Minf[0]>1.0 else 0.3)
# for c in Nicholson2024.keys():
#     for xi in range(len(Nicholson2024[c].x)):
#         ax.semilogx(Nicholson2024[c].yplusVD[xi], Nicholson2024[c].uplusVD[xi], '-',
#                 color=col(getattr(Nicholson2024[c],Cquant)[xi]), label=f"N24:{c}")
ax.semilogx(Sillero2014.yplus[0], Sillero2014.uplus[0], color='black', 
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
for c in Trettel2016.keys():
    ax.semilogx(Trettel2016[c].yplusTL[0], Trettel2016[c].uplusTL[0], 
                color=col(getattr(Trettel2016[c],Cquant)), label=f"TL16:{c}")
for c in Modesti2016.keys():
    ax.semilogx(Modesti2016[c].yplusTL[0], Modesti2016[c].uplusTL[0], 
                color=col(getattr(Modesti2016[c],Cquant)), label=f"M16:{c}")
# for c in Zhang2018.keys():
#     ax.semilogx(Zhang2018[c].yplusTL[0], Zhang2018[c].uplusTL[0], 
#                 color=col(getattr(Zhang2018[c],Cquant)), label=f"Z18:{c}")
# for c in Volpiani2020.keys():
#     ax.semilogx(Volpiani2020[c].yplusTL[Vxs], Volpiani2020[c].uplusTL[Vxs], '--',
#                 color=col(getattr(Volpiani2020[c],Cquant)[Vxs]), label=f"V20:{c}")
# for c in Wenzel2019.keys():
#     # xi = np.where(Wenzel2019[c].Retau>=Retau_plt)[0][0]
#     lstyle = '-' if Wenzel2019[c].Minf[0]>1.0 else '--'
#     for xi in range(0,len(Wenzel2019[c].x),len(Wenzel2019[c].x)//6):
#         ax.semilogx(Wenzel2019[c].yplusTL[xi], Wenzel2019[c].uplusTL[xi], lstyle,
#                     color=col(getattr(Wenzel2019[c],Cquant)), label=f"W19:{c}",
#                     alpha=1.0 if Wenzel2019[c].Minf[0]>1.0 else 0.3)
# for c in Nicholson2024.keys():
#     for xi in range(len(Nicholson2024[c].x)):
#         ax.semilogx(Nicholson2024[c].yplusTL[xi], Nicholson2024[c].uplusTL[xi], '-',
#                 color=col(getattr(Nicholson2024[c],Cquant)[xi]), label=f"N24:{c}")
ax.semilogx(Sillero2014.yplus[0], Sillero2014.uplus[0], color='black', 
            label="Sillero et al. (2014)")
# ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^*$")
ax.set_ylabel(r"$\overline{u}^+_{\mathrm{TL}}$")
ax.set_title(r"Trettel \& Larsson Correction")
cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(Cquant)
fig.tight_layout(); fig.show()

fig, ax = plt.subplots(figsize=[6,4])
ax.grid(which='both', color='0.9')
for c in Trettel2016.keys():
    ax.semilogx(Trettel2016[c].yplusV[0], Trettel2016[c].uplusV[0], 
                color=col(getattr(Trettel2016[c],Cquant)), label=f"TL16:{c}")
for c in Modesti2016.keys():
    ax.semilogx(Modesti2016[c].yplusV[0], Modesti2016[c].uplusV[0], 
                color=col(getattr(Modesti2016[c],Cquant)), label=f"M16:{c}")
# for c in Zhang2018.keys():
#     ax.semilogx(Zhang2018[c].yplusV[0], Zhang2018[c].uplusV[0], 
#                 color=col(getattr(Zhang2018[c],Cquant)), label=f"Z18:{c}")
# for c in Volpiani2020.keys():
#     ax.semilogx(Volpiani2020[c].yplusV[Vxs], Volpiani2020[c].uplusV[Vxs], '--', 
#                 color=col(getattr(Volpiani2020[c],Cquant)[Vxs]), label=f"V20:{c}")
# for c in Wenzel2019.keys():
#     # xi = np.where(Wenzel2019[c].Retau>=Retau_plt)[0][0]
#     lstyle = '-' if Wenzel2019[c].Minf[0]>1.0 else '--'
#     for xi in range(0,len(Wenzel2019[c].x),len(Wenzel2019[c].x)//6):
#         ax.semilogx(Wenzel2019[c].yplusV[xi], Wenzel2019[c].uplusV[xi], lstyle,
#                     color=col(getattr(Wenzel2019[c],Cquant)), label=f"W19:{c}",
#                     alpha=1.0 if Wenzel2019[c].Minf[0]>1.0 else 0.3)
# for c in Nicholson2024.keys():
#     for xi in range(len(Nicholson2024[c].x)):
#         ax.semilogx(Nicholson2024[c].yplusV[xi], Nicholson2024[c].uplusV[xi], '-',
#                 color=col(getattr(Nicholson2024[c],Cquant)[xi]), label=f"N24:{c}")
ax.semilogx(Sillero2014.yplus[0], Sillero2014.uplus[0], color='black', 
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
for c in Trettel2016.keys():
    ax.semilogx(Trettel2016[c].yplusGFM[0], Trettel2016[c].uplusGFM[0], 
                color=col(getattr(Trettel2016[c],Cquant)), label=f"TL16:{c}")
for c in Modesti2016.keys():
    ax.semilogx(Modesti2016[c].yplusGFM[0], Modesti2016[c].uplusGFM[0], 
                color=col(getattr(Modesti2016[c],Cquant)), label=f"M16:{c}")
# for c in Zhang2018.keys():
#     ax.semilogx(Zhang2018[c].yplusGFM[0], Zhang2018[c].uplusGFM[0], 
#                 color=col(getattr(Zhang2018[c],Cquant)), label=f"Z18:{c}")
# for c in Volpiani2020.keys():
#     ax.semilogx(Volpiani2020[c].yplusGFM[Vxs], Volpiani2020[c].uplusGFM[Vxs], '--', 
#                 color=col(getattr(Volpiani2020[c],Cquant)[Vxs]),label=f"V20:{c}")
# for c in Wenzel2019.keys():
#     # xi = np.where(Wenzel2019[c].Retau>=Retau_plt)[0][0]
#     lstyle = '-' if Wenzel2019[c].Minf[0]>1.0 else '--'
#     for xi in range(0,len(Wenzel2019[c].x),len(Wenzel2019[c].x)//6):
#         ax.semilogx(Wenzel2019[c].yplusGFM[xi], Wenzel2019[c].uplusGFM[xi], lstyle,
#                     color=col(getattr(Wenzel2019[c],Cquant)), label=f"W19:{c}",
#                     alpha=1.0 if Wenzel2019[c].Minf[0]>1.0 else 0.3)
# for c in Nicholson2024.keys():
#     for xi in range(len(Nicholson2024[c].x)):
#         ax.semilogx(Nicholson2024[c].yplusGFM[xi], Nicholson2024[c].uplusGFM[xi], '-',
#                 color=col(getattr(Nicholson2024[c],Cquant)[xi]), label=f"N24:{c}")
ax.semilogx(Sillero2014.yplus[0], Sillero2014.uplus[0], color='black', 
            label="Sillero et al. (2014)")
# ax.legend(loc='best', framealpha=1)
ax.set_xlabel(r"$y^*$")
ax.set_ylabel(r"$\overline{u}^+_{\mathrm{GFM}}$")
ax.set_title(r"Griffin, Fu, \& Moin Correction")
cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(Cquant)
fig.tight_layout(); fig.show()

# %%
