# %% Imports and plot formatting
import numpy as np
import IT_Pi
from glob import glob

import matplotlib.pyplot as plt
from matplotlib import cm, colors, rcParams
rcParams.update({'text.usetex': True, 'font.size': 14, 'font.family': 'serif'})

from database import db_config as db
import database.filepaths as fpaths
gonzalo_taudata = fpaths.db_path + "/../GonzaloData/tau_data.npz"

# %% Script settings/inputs
Nk = 30  # Number of times to repeat the training
Njobs = -1  # Number of parallel jobs (-1 for all cores)
Ntrain = int(2e3)  # Number of training points
Vars = ['Ue', 'd1', 'rho', 'd2', 'mu', 'dPe']  # Variables to consider when building Pi groups
output_path = "./npzs"

# %% Create data class from abstract base
# PiY = tau_w/(rho*U_e^2)
class tau_data(IT_Pi.ITPi_data):
    _vars_all = ['Ue', 'delta', 'd1', 'd2',
                      'mu', 'dPe', 'dPw', 'rho']
    _D_in = np.matrix([
        [ 1, 1, 1, 1, -1, -2, -2, -3],  # L
        [ 0, 0, 0, 0,  1,  1,  1,  1],  # M
        [-1, 0, 0, 0, -1, -2, -2,  0],  # t
    ])

    def extract_vars(self, cases:list[db.BL]):
        # TODO: Finish putting this together
        X, Y = np.ndarray([0,self._lvars]), np.ndarray([0,1])
        for c in cases:
            assert np.all(c.hasdata(('uinf','muinf','rhoinf','Minf',
                                     'delta99','delta1','delta2','delta1k',
                                     'Bk','Cf','tauw')))
            Xc = np.hstack((c.uinf, c.delta99, c.delta1, c.delta2, c.muinf/c.rhoinf))
            Yc = 0.5*c.Cf
            X, Y = np.vstack((X, Xc)), np.vstack((Y, Yc))
        return (X, Y)

# %% Load in data and preprocess

# Read in incompressible PG data from Gonzalo
GonzaloDat = np.load(gonzalo_taudata)
npoints = GonzaloDat['X'].shape[0]
PiY = GonzaloDat['Y'] / GonzaloDat['X'][:,[0]]**2
# Pop out second to last column (kinetic energy metric \int_0^\delta U^2dy I'm not bothering with)
X = np.delete(GonzaloDat['X'],-2,axis=1)
# rho=1 to X to make compatible with compressible data
X = np.hstack((X, np.ones((npoints,1))))

# Build data object
dat = tau_data(X, PiY)
ivars = [dat._vars_all.index(x) for x in Vars]

# %% Run IT_Pi
if __name__ == "__main__":  # Necessary for parallelization
    for k in range(Nk):
        print(f'Iteration = {k+1}')
        for num_input in (1, 2):
            X, Y, basis_matrices = dat.get_traindata(Vars)

            results = IT_Pi.run(
                X, Y,
                basis_matrices,
                num_input=num_input,
                estimator="kraskov",
                estimator_params={"k": 5},
                n_jobs=Njobs,
                optimize_output=False,
            )

            fnm = f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{num_input}_{k}_output'
            np.savez(fnm, **results)

    # Plot some results
    # Visualize exponents
    e, MI = [], []
    fig, axs = plt.subplots(2, 1, figsize=(8,8), sharex=True, layout='constrained')
    for Ni in [1,2]:
        fnm = f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_*_output.npz'
        flist = glob(fnm)
        e.append(np.zeros((len(flist), Ni, len(Vars))))
        MI.append(np.zeros(len(flist)))
        for i,f in enumerate(flist):
            data = np.load(f, allow_pickle=True)

            # Normalize exponents and remove small terms (adjusting rest to preserve
            # non-dimensionality)
            eorig = IT_Pi.norme(data['input_coef'], 5)
            # flip eorig so dPe ~ 0 is always the first vector 
            # - SH: I think this is just so first Pi group doesn't have dPe?
            eorig = np.roll(eorig, eorig[:, Vars.index('dPe')].argmin(), axis=0)
            for k, e_ in enumerate(eorig):
                inds = np.where(np.abs(eorig[k]) < .01)[0]
                e[-1][i,k] = IT_Pi.return_enew(np.array(tau_data._D_in[:,ivars]), e_, inds)

            MI[-1][i] = data['optimized_MI']
        Cmin, Cmax = np.min(MI[-1]), np.max(MI[-1])
        col = lambda q: cm.rainbow((q-Cmin)/(Cmax-Cmin))
        cmap = cm.ScalarMappable(colors.Normalize(vmin=Cmin, vmax=Cmax),"rainbow")
        ax = axs[Ni-1]
        ax.grid(True, which='both', alpha=0.5)
        for k in range(len(MI[-1])):
            ax.plot(range(len(Vars)), e[-1][k,0], '-o', color=col(MI[-1][k]))
            if Ni>1:
                ax.plot(range(len(Vars)), e[-1][k,1], '--o', color=col(MI[-1][k]))
        cbar = fig.colorbar(cmap,ax=ax); cbar.set_label("MI",rotation=0,labelpad=15)
    _ = ax.set_xticks(range(len(Vars)), Vars)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_exp.pdf')
    fig.show()

    # Plot Pi groups against each other
    X, PiY, _ = dat.get_data(Vars)
    # Ni = 1
    PiX = np.array([IT_Pi.getPiIfromXe(X, e_.squeeze()) for e_ in e[0]])
    kmin = MI[0].argmin()
    fig, ax = plt.subplots(figsize=(6,6), layout='constrained')
    ax.plot(PiX[kmin], PiY, 'o', color='k', markersize=2,alpha=0.75)
    ax.set_xlabel(r"$\Pi_1$")
    ax.set_ylabel(r"$\Pi_\tau$", rotation=0)
    fig.show()
    # Ni = 2
    PiX2 = np.array([[IT_Pi.getPiIfromXe(X, e_) for e_ in ek] for ek in e[1]])
    kmin = MI[1].argmin()
    Cmin, Cmax = np.min(PiY), np.max(PiY)
    cmap = cm.ScalarMappable(colors.Normalize(vmin=Cmin*1e3, vmax=Cmax*1e3),"rainbow")
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    ax.scatter(PiX2[kmin,0], PiX2[kmin,1], c=PiY*1e3, cmap='rainbow', edgecolors='k')
    cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(r"$\Pi_\tau\times10^3$",labelpad=15)
    ax.set_xlabel(r"$\Pi_1$")
    ax.set_ylabel(r"$\Pi_2$", rotation=0)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_space.pdf')
    fig.show()

# %%
