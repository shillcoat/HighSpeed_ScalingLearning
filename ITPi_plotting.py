import matplotlib.pyplot as plt
from matplotlib import cm, colors
import IT_Pi
import numpy as np
from ITPi_classes import tau_data

__all__ = [
    "plt_exps",
    "plt_1Pi",
    "plt_2Pi",
]
lstyles = ['-', '--', ':', '-.']

def plt_exps(npzs, Ni, Vars, Varlbls=None, ax=None, exp_thresh=0.01):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5), layout='constrained')
    else: fig = ax.get_figure()
    ivars = [tau_data._vars_all.index(x) for x in Vars]
    if Varlbls is None:
        Varlbls = Vars

    e = np.array([IT_Pi.get_exp(npz,tau_data._D_in[:,ivars],Vars,exp_thresh) for npz in npzs])
    MI = np.array([np.load(npz,allow_pickle=True)['optimized_MI'] for npz in npzs])
    Cmin, Cmax = np.min(MI), np.max(MI)
    
    norm = colors.Normalize(vmin=Cmin, vmax=Cmax)
    cmap = cm.ScalarMappable(norm=norm, cmap='rainbow')
    col = lambda q: cmap.to_rgba(q)

    for k in range(len(MI)):
        if Ni == 1:
            ax.plot(range(len(Vars)), e[k,0], '-o', color=col(MI[k]), label=f"MI={MI[k]:.3f}")
        else:
            for i in range(Ni):
                ax.plot(range(len(Vars)), e[k,i], lstyles[i%len(lstyles)]+'o', color=col(MI[k]))
    ax.set_ylabel("exponent")

    cbar = fig.colorbar(cmap,ax=ax); cbar.set_label("MI",rotation=0,labelpad=15)
    _ = ax.set_xticks(range(len(Vars)), [f"${v}$" for v in Varlbls])
    ax.grid(True, which='both', alpha=0.5)
    return fig, ax, e, MI

def plt_1Pi(X, PiY, e, Vars, Varlbls=None, PiYlbl=None, ax=None, colQ=None, colLbl=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5), layout='constrained')
    else: fig = ax.get_figure()
    if Varlbls is None:
        Varlbls = Vars
    if PiYlbl is None:
        PiYlbl = r"$\Pi_o$"

    PiX = np.array(IT_Pi.getPiIfromXe(X, e.squeeze()))
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

def plt_2Pi(X, PiY, e, Vars, Varlbls=None, PiYlbl=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    else: fig = ax.get_figure()
    if Varlbls is None:
        Varlbls = Vars
    if PiYlbl is None:
        PiYlbl = r"$\Pi_o$"

    PiX = np.array([IT_Pi.getPiIfromXe(X, e_) for e_ in e]).T
    Cmin, Cmax = np.min(PiY), np.max(PiY)
    cmap = cm.ScalarMappable(colors.Normalize(vmin=Cmin, vmax=Cmax),"rainbow")
    ax.scatter(PiX[:,0], PiX[:,1], c=PiY, cmap='rainbow', **kwargs)
    cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(PiYlbl, labelpad=15)
    ax.set_xlabel(r"$\Pi_1$")
    ax.set_ylabel(r"$\Pi_2$", rotation=0)
    return fig, ax