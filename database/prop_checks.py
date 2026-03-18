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

import db_config as db
import filepaths as fpath

def check_edges(dat:db.BL, edges:list[str]=None, sigma_smooth:list[float]|str=None,title:str=None,interp:bool=True):
    if edges is None:
        edges = ['delta99', 'delta1', 'delta2', 'delta1k', 'delta2k']
    for e in edges:
        if dat.hasdata(e):
            dval = getattr(dat, e)
            dcalc, _ = dat.find_edge(e, sigma_smooth=sigma_smooth, interp=interp)
            plt.figure()
            plt.scatter(dat.x, dval, label='Precomputed', marker='.', s=2)
            plt.scatter(dat.x, dcalc, label='Calculated', marker='.', s=2)
            plt.legend()
            plt.xlabel('x')
            plt.ylabel(e)
            plt.ylim(0, 1.2*max(dval))
            if title is not None:
                plt.title(title)
            plt.show()

# %% Run
if __name__ == "__main__":
    # Select case to run on
    cases = glob(f"{fpath.Volpiani2020_path}/*.dill")
    c = cases[0]
    print(c.split('/')[-1])

    # Compute a bunch of BL thicknesses
    dat = db.load_case(c)
    if dat.v is None:
        dat.v = np.zeros_like(dat.u)
    if dat.u.ndim > 1:
        nx, ny = dat.u.shape
    else:
        nx, ny = 1, dat.u.shape[0]

    dx = np.mean(np.diff(dat.x)) if nx > 1 else 1
    ssmooth = [8, 2] # Helpful smoothing kernel for Volpiani
    # ssmooth = None

    d99, id99 = dat.find_edge('delta99', sigma_smooth=ssmooth)
    dinf, idinf = dat.find_edge('deltainf', sigma_smooth=ssmooth)
    d1, id1 = dat.find_edge('delta1', sigma_smooth=ssmooth)
    d1k, id1k = dat.find_edge('delta1k', sigma_smooth=ssmooth)
    d2, id2 = dat.find_edge('delta2', sigma_smooth=ssmooth)
    d2k, id2k = dat.find_edge('delta2k', sigma_smooth=ssmooth)

    d99GFM, id99GFM = dat.find_edge('delta99GFM')

    # Check BL thickness calculations vs precomputed values (if available)
    check_edges(dat, sigma_smooth=ssmooth)

    # Plot some properties midway through the domain (streamwise)
    for prop in ['u','rho','P','T']:
        i = nx//2 if nx > 1 else -1
        p = getattr(dat,prop)
        p = p[i] if p.ndim > 1 else p
        pinf = getattr(dat,prop+'inf')
        if pinf is None:
            pinf = p[idinf[i]]
        pinf = pinf[i] if pinf.size > 1 else pinf

        plt.figure()
        plt.plot(dat.y,p)
        plt.vlines([d99[i],d99GFM[i], dinf[i]],p.min(),p.max(),colors='k',linestyles=['-', '-.','--'])
        plt.vlines([d1[i],d2[i]],p.min(),p.max(),colors='C7',alpha=0.75,linestyles=['-','--'])
        plt.vlines([d1k[i],d2k[i]],p.min(),p.max(),colors='C3',alpha=0.75,linestyles=['-.',':'])
        plt.hlines(pinf,dat.y[0],dat.y[-1],colors='k', linestyles=':')
        proxylines = plt.plot([],[], 'k-',
                            [],[], 'k-.',
                            [],[], 'k--',
                            [],[], 'k:',
                            [],[], '-C7',
                            [],[], '--C7',    
                            [],[], '-.C3',
                            [],[], ':C3',
                            )
        plt.xlabel("y")
        plt.ylabel(prop)
        plt.grid()
        plt.legend(proxylines, ['d99','d99GFM','dinf',f'{prop}inf','d1','d2','d1k','d2k'], loc='lower right')
        plt.show()

        plt.figure(figsize=(8,4))
        plt.pcolormesh(*np.meshgrid(dat.x, dat.y), getattr(dat,prop).T, shading='auto', cmap=cm.bwr)
        plt.ylim(0,dat.y[np.max(idinf)])
        plt.colorbar(label=prop)
        plt.gca().set_aspect('auto')
        plt.xlabel("x")
        plt.ylabel("y")
# %%
