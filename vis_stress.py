# %% Imports and plot formatting
import numpy as np
from glob import glob

import matplotlib.pyplot as plt
from matplotlib import rcParams, cm, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
rcParams.update({'text.usetex': True, 
                 'font.size': 14, 
                 'font.family': 'serif'})

from database import db_config as db
import database.filepaths as fpaths

import pdb

def mach_linestyle(M):
    if abs(M - 0.7) < 0.1:   return '-'
    elif abs(M - 1.7) < 0.1: return '--'
    elif abs(M - 3.0) < 0.1: return '-.'
    else:                     return ':'

# %% Load Trettel2016 cases and plot viscous and Reynolds stresses, coloured by Retau
cases = glob(f"{fpaths.Trettel2016_path}/*.dill")
cases = sorted(cases, key=lambda c: float(db.load_case(c).Retau))
Re_values = [db.load_case(c).Retau for c in cases]
norm = colors.Normalize(vmin=min(Re_values), vmax=max(Re_values))
cmap = cm.rainbow

trettelTauTot = []

fig, [axV, axRe] = plt.subplots(1,2, figsize=(12,5))

# Create inset axes
axV_inset = inset_axes(axV, width="50%", height="70%", loc='upper right')
axRe_inset = inset_axes(axRe, width="50%", height="70%", loc='upper right')

for c, Re in zip(cases, Re_values):
    cname = c.split('/')[-1][:-5]
    cs = db.load_case(c)
    color = cmap(norm(Re))
    ls = mach_linestyle(cs.Mbulk)
    tauRe = -cs.ruppvpp[0]/cs.tauw
    tauV = (cs.mu * np.gradient(cs.u, cs.y, axis=-1) / cs.tauw).flatten()

    axV.plot(cs.y, tauV, color=color, label=cname, linestyle=ls)
    axRe.plot(cs.y, tauRe, color=color, label=cname, linestyle=ls)

    trettelTauTot.append({'c': cs, 'tau': np.vstack((cs.y, tauV+tauRe))})
    # Plot inset region
    mask = cs.y < 0.15
    axV_inset.plot(cs.y[mask], tauV[mask], color=color, linestyle=ls)
    axRe_inset.plot(cs.y[mask], tauRe[mask], color=color, linestyle=ls)

# Format insets
for ax_inset in [axV_inset, axRe_inset]:
    ax_inset.set_xlim(0, 0.15)
    ax_inset.tick_params(labelsize=9)

# Mark inset region on main axes
mark_inset(axV, axV_inset, loc1=2, loc2=3, fc="none", ec="0.5")
mark_inset(axRe, axRe_inset, loc1=2, loc2=3, fc="none", ec="0.5")

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=[axV, axRe], label=r"$Re_{\tau}$")

axV.set_xlabel(r"$y$")
axV.set_ylabel(r"$\tau_\nu/\tau_w$")
axRe.set_xlabel(r"$y$")
axRe.set_ylabel(r"$\tau_R/\tau_w$")

# %% Load Volpiani cases and plot viscous and Reynolds stresses, coloured by Retau
cases = glob(f"{fpaths.Volpiani2020_path}/*.dill")
ix = 500
cases = sorted(cases, key=lambda c: float(db.load_case(c).Retau[ix]))
Re_values = [db.load_case(c).Retau[ix] for c in cases]
norm = colors.Normalize(vmin=min(Re_values), vmax=max(Re_values))
cmap = cm.rainbow

volpianiTauTot = []

fig, [axV, axRe] = plt.subplots(1,2, figsize=(12,5))

# Create inset axes
axV_inset = inset_axes(axV, width="50%", height="70%", loc='upper right')
axRe_inset = inset_axes(axRe, width="50%", height="70%", loc='upper right')

for c, Re in zip(cases, Re_values):
    cname = c.split('/')[-1][:-5]
    cs = db.load_case(c)
    ls = '-' if np.abs(cs.Minf-2.28)<0.2 else '--'
    _, id99 = cs.find_edge(edge_type='delta99')
    id99 = id99[ix]
    color = cmap(norm(Re))
    tauRe = -cs.ruppvpp[ix,:id99+1]/cs.tauw[ix]
    tauV = (cs.mu[ix,:id99+1] * np.gradient(cs.u[ix,:id99+1], cs.y[:id99+1], axis=-1) / cs.tauw[ix]).flatten()

    axV.plot(cs.y[:id99+1], tauV, color=color, label=cname, linestyle=ls)
    axRe.plot(cs.y[:id99+1], tauRe, color=color, label=cname, linestyle=ls)

    volpianiTauTot.append({'c': cs, 'tau': np.vstack((cs.y[:id99+1], tauV+tauRe))})
    # Plot inset region
    mask = cs.yplus[ix] < 100
    axV_inset.plot(cs.y[mask], tauV[mask[:id99+1]], color=color, linestyle=ls)
    axRe_inset.plot(cs.y[mask], tauRe[mask[:id99+1]], color=color, linestyle=ls)

# Format insets
for ax_inset in [axV_inset, axRe_inset]:
    ax_inset.set_xlim(0, cs.y[mask].max())
    ax_inset.tick_params(labelsize=9)

# Mark inset region on main axes
mark_inset(axV, axV_inset, loc1=2, loc2=3, fc="none", ec="0.5")
mark_inset(axRe, axRe_inset, loc1=2, loc2=3, fc="none", ec="0.5")

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=[axV, axRe], label=r"$Re_{\tau}$")

axV.set_xlabel(r"$y$")
axV.set_ylabel(r"$\tau_\nu/\tau_w$")
axRe.set_xlabel(r"$y$")
axRe.set_ylabel(r"$\tau_R/\tau_w$")

# %% Compare total stresses
fig, ax = plt.subplots(figsize=(6,5))
for c in trettelTauTot:
    cs = c['c']
    y = c['tau'][0,:]
    tauTot = c['tau'][1,:]
    ax.plot(y, tauTot, linestyle=mach_linestyle(cs.Mbulk), c='k')
for c in volpianiTauTot:
    cs = c['c']
    y = c['tau'][0,:]
    tauTot = c['tau'][1,:]
    ls = '-' if np.abs(cs.Minf-2.28)<0.2 else '--'
    ax.plot(y/np.max(y), tauTot, linestyle=ls, c='r')
ax.set_ylabel(r'$\tau_{tot}/\tau_w$')
ax.set_xlabel(r'$y/\delta$')
dummyTrettel = ax.plot([],[],c='k', label='Trettel2016')
dummyVolpiani = ax.plot([],[],c='r', label='Volpiani2020')
ax.legend(loc='lower left')
# %%
