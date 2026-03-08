import matplotlib.pyplot as plt
from matplotlib import cm, colors
import IT_Pi
import numpy as np

import pdb

__all__ = [
    "plt_exps",
    "plt_1Pi",
    "plt_2Pi",
]
lstyles = ['-', '--', ':', '-.']

def plt_exps(results, Ni, dat_obj, Vars, Varlbls=None, ax=None, exp_thresh=0.01, inorm=None):
    # Currently only supports a single output Pi group (if optimizing output)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5), layout='constrained')
    else: fig = ax.get_figure()
    ivars = [dat_obj._vars_all.index(x) for x in Vars]
    if Varlbls is None:
        Varlbls = Vars

    e_in, e_out = [], []
    for res in results:
        ei, eo = IT_Pi.get_exp(res,dat_obj._D_in[:,ivars],Vars,exp_thresh,inorm=inorm)
        e_in.append(ei)
        if eo is not None: e_out.append(eo)
    e_in = np.array(e_in)
    e_out = np.array(e_out) if e_out else None

    MI = []
    for res in results:
        if isinstance(res,str):
            res = np.load(res, allow_pickle=True)
        MI.append(res['optimized_MI'])
    MI = np.array(MI)
    # MI = np.array([np.load(res,allow_pickle=True)['optimized_MI'] for res in results])
    Cmin, Cmax = np.min(MI), np.max(MI)
    
    norm = colors.Normalize(vmin=Cmin, vmax=Cmax)
    cmap = cm.ScalarMappable(norm=norm, cmap='rainbow')
    col = lambda q: cmap.to_rgba(q)

    for k in range(len(MI)):
        if Ni == 1:
            ax.plot(range(len(Vars)), e_in[k,0], '-o', color=col(MI[k]))
            if e_out is not None:
                ax.scatter(range(len(Vars)), e_out[k,0], color=col(MI[k]), edgecolor='k')
        else:
            for i in range(Ni):
                ax.plot(range(len(Vars)), e_in[k,i], lstyles[i%len(lstyles)]+'o', color=col(MI[k]))
            if e_out is not None:
                ax.scatter(range(len(Vars)), e_out[k,0], color=col(MI[k]), edgecolor='k')
    ax.set_ylabel("exponent")

    cbar = fig.colorbar(cmap,ax=ax); cbar.set_label("MI",rotation=0,labelpad=15)
    _ = ax.set_xticks(range(len(Vars)), [f"${v}$" for v in Varlbls])
    ax.grid(True, which='both', alpha=0.5)
    return fig, ax, e_in, e_out, MI

def plt_1Pi(X, Y, e_in, e_out, PiYlbl=None, ax=None, colQ=None, colLbl=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5), layout='constrained')
    else: fig = ax.get_figure()
    if PiYlbl is None:
        PiYlbl = r"$\Pi_o$"

    PiX = np.array(IT_Pi.getPiIfromXe(X, e_in.squeeze()))
    PiY = Y
    if e_out is not None:
        PiY = Y * np.array(IT_Pi.getPiIfromXe(X, e_out.squeeze()))[:,np.newaxis]

    if colQ is None:
        ax.scatter(PiX, PiY.squeeze(), c='k', **kwargs)
    else:
        Cmin, Cmax = np.min(colQ), np.max(colQ)
        cmap = cm.ScalarMappable(colors.Normalize(vmin=Cmin, vmax=Cmax),"rainbow")
        ax.scatter(PiX, PiY.squeeze(), c=colQ, cmap='rainbow', **kwargs)
        cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(colLbl, labelpad=15)
    ax.set_xlabel(r"$\Pi_1$")
    ax.set_ylabel(PiYlbl, rotation=0)
    return fig, ax

def plt_2Pi(X, Y, e_in, e_out, PiYlbl=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    else: fig = ax.get_figure()
    if PiYlbl is None:
        PiYlbl = r"$\Pi_o$"

    PiX = np.array([IT_Pi.getPiIfromXe(X, e_) for e_ in e_in]).T
    PiY = Y
    if e_out is not None:
        PiY = Y * np.array(IT_Pi.getPiIfromXe(X, e_out.squeeze()))[:,np.newaxis]
    Cmin, Cmax = np.min(PiY), np.max(PiY)
    cmap = cm.ScalarMappable(colors.Normalize(vmin=Cmin, vmax=Cmax),"rainbow")
    ax.scatter(PiX[:,0], PiX[:,1], c=PiY, cmap='rainbow', **kwargs)
    cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(PiYlbl, labelpad=15)
    ax.set_xlabel(r"$\Pi_1$")
    ax.set_ylabel(r"$\Pi_2$", rotation=0)
    return fig, ax