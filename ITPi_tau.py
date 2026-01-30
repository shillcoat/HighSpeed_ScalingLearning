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
Nk = 10  # Number of times to repeat the training
Njobs = -1  # Number of parallel jobs (-1 for all cores)
Ntrain = int(5e3)  # Number of training points
Vars = ['Ue', 'd1', 'rhoe', 'd2', 'mue', 'dPe', 'ae']  # Variables to consider when building Pi groups. 
# Note the order of Vars DOES matter in so much as first k must contain all dimensions, where k is the
# number of distinct dimensions (due to way calc_basis function works)
output_path = "./npzs_comp"
exp_thresh = 0.01  # Threshold for setting an exponent to 0

# %% Create data class from abstract base
# PiY = tau_w/(rho*U_e^2)
# TODO: Maybe try giving it delta1k too to see if can rebuild Rotta-Clauser parameter identified by 
# Wenzel as appropriate self-similarity parameter
class tau_data(IT_Pi.ITPi_data):
    _vars_all = ['Ue', 'delta', 'd1', 'd2',
                      'mue', 'dPe', 'dPw', 'rhoe', 'ae']
    _D_in = np.matrix([
        [ 1, 1, 1, 1, -1, -2, -2, -3,  1],  # L
        [ 0, 0, 0, 0,  1,  1,  1,  1,  0],  # m
        [-1, 0, 0, 0, -1, -2, -2,  0, -1],  # t
    ])

    def extract_vars(self, cases:list[db.BL]):
        # TODO: Finish putting this together
        X, Y = np.ndarray([0,self._lvars]), np.ndarray([0,1])
        for c in cases:
            assert np.all(c.hasdata(('uinf','muinf','rhoinf','Minf',
                                     'delta99','delta1','delta2','delta1k',
                                     'Bk','Cf','tauw','P')))
            if c.P.ndim > 1:
                ide = np.argmin(np.abs(c.y-c.delta99[:,np.newaxis]),axis=-1)
                dPe = np.gradient(c.P[range(len(c.x)),ide],c.x)
                dPw = np.gradient(c.P[...,0],c.x)
            else: 
                dPe = dPw = c.Bk*c.tauw/c.delta1k
            Xc = np.vstack((c.uinf, c.delta99, c.delta1, c.delta2, c.muinf, dPe, dPw, c.rhoinf, c.uinf/c.Minf)).T
            Yc = 0.5*c.Cf[:,np.newaxis]
            X, Y = np.vstack((X, Xc)), np.vstack((Y, Yc))
        self.append_data(X,Y)
        return (X, Y)

# %% Load in data and preprocess

# Read in incompressible PG data from Gonzalo
GonzaloDat = np.load(gonzalo_taudata)
npoints = GonzaloDat['X'].shape[0]
PiY = GonzaloDat['Y'] / GonzaloDat['X'][:,[0]]**2
# Pop out second to last column (kinetic energy metric \int_0^\delta U^2dy I'm not bothering with)
X = np.delete(GonzaloDat['X'],-2,axis=1)
# Add rho=1, a=U/0.01 (M=0.01) to X to make compatible with compressible data
X = np.hstack((X, [1, 1e2]*np.ones((npoints,2))))

# Build data object
# # Cut out some of Gonzalo's data for now so mostly compressible data
# dat = tau_data(X[::5], PiY[::5])
# Try just compressible data to see what happens:
dat = tau_data()
ivars = [dat._vars_all.index(x) for x in Vars]

# Get Wenzel data
cases = []
for cs in glob(f"{fpaths.Wenzel2019_path}/*.dill"):
    cases.append(db.load_case(cs))
_ = dat.extract_vars(cases)

# %% Run IT_Pi
if __name__ == "__main__":  # Necessary for parallelization
    for k in range(Nk):
        print(f'Iteration = {k+1}')
        for num_input in (1, 2):
            X, Y, basis_matrices = dat.get_traindata(Vars, Ntrain=Ntrain)

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
    fig, axs = plt.subplots(1, 2, figsize=(12,5), sharex=True, layout='constrained')
    for Ni in [1,2]:
        fnm = f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_*_output.npz'
        flist = glob(fnm)
        e.append(np.array(
            [IT_Pi.get_exp(npz,tau_data._D_in[:,ivars],Vars,exp_thresh) for npz in flist]))
        MI.append(np.array(
            [np.load(npz,allow_pickle=True)['optimized_MI'] for npz in flist]))

        Cmin, Cmax = np.min(MI[-1]), np.max(MI[-1])
        col = lambda q: cm.rainbow((q-Cmin)/(Cmax-Cmin))
        cmap = cm.ScalarMappable(colors.Normalize(vmin=Cmin, vmax=Cmax),"rainbow")
        ax = axs[Ni-1]
        ax.grid(True, which='both', alpha=0.5)
        for k in range(len(MI[-1])):
            ax.plot(range(len(Vars)), e[-1][k,0], '-o', color=col(MI[-1][k]))
            if Ni>1:
                ax.plot(range(len(Vars)), e[-1][k,1], '--o', color=col(MI[-1][k]))
            ax.set_ylabel("exponent")
        cbar = fig.colorbar(cmap,ax=ax); cbar.set_label("MI",rotation=0,labelpad=15)
    _ = ax.set_xticks(range(len(Vars)), Vars)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_exp.pdf')
    fig.show()

    # Plot Pi groups against each other
    X, PiY, _ = dat.get_data(Vars)
    # Ni = 1
    PiX = np.array([IT_Pi.getPiIfromXe(X, e_.squeeze()) for e_ in e[0]])
    Me = X[:,0]/X[:,-1]
    kmin = MI[0].argmin()
    print(f"Optimal exponents, Ni=1: \n{e[0][kmin].squeeze()}")
    fig, ax = plt.subplots(figsize=(6,6), layout='constrained')
    Cmin, Cmax = np.min(Me), np.max(Me)
    cmap = cm.ScalarMappable(colors.Normalize(vmin=Cmin, vmax=Cmax),"rainbow")
    ax.scatter(PiX[kmin], PiY, c=Me, cmap='rainbow', s=2, alpha=0.75)
    cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(r"$M_e$",labelpad=15)
    ax.set_xlabel(r"$\Pi_1$")
    ax.set_ylabel(r"$\Pi_\tau$", rotation=0)
    fig.show()
    # Ni = 2
    PiX2 = np.array([[IT_Pi.getPiIfromXe(X, e_) for e_ in ek] for ek in e[1]])
    kmin = MI[1].argmin()
    print(f"Optimal exponents, Ni=2: \n{e[1][kmin].squeeze()}")
    Cmin, Cmax = np.min(PiY), np.max(PiY)
    cmap = cm.ScalarMappable(colors.Normalize(vmin=Cmin*1e3, vmax=Cmax*1e3),"rainbow")
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    ax.scatter(PiX2[kmin,0], PiX2[kmin,1], c=PiY*1e3, cmap='rainbow')  # , edgecolors='k')
    cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(r"$\Pi_\tau\times10^3$",labelpad=15)
    ax.set_xlabel(r"$\Pi_1$")
    ax.set_ylabel(r"$\Pi_2$", rotation=0)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_space.pdf')
    fig.show()

# %%
