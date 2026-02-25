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

from ITPi_classes import tau_data

# %% Script settings/inputs
Nk = 10  # Number of times to repeat the training
Njobs = -1  # Number of parallel jobs (-1 for all cores)
Ntrain = int(5e3)  # Number of training points
Vars = ['Ue', 'd1', 'rhoe', 'd2', 'mue', 'dPe', 'muw', 'rhow']  # Variables to consider when building Pi groups.
# LaTeX labels for variables (in same order as Vars)
Varlbls = [r"U_e", r"\delta_1", r"\rho_e", r"\delta_2", r"\mu_e", r"dP_e", r"\mu_w", r"\rho_w"]
# Note the order of Vars DOES matter in so much as first k must contain all dimensions, where k is the
# number of distinct dimensions (due to way calc_basis function works)
output_path = "./npzs_comp"
exp_thresh = 0.01  # Threshold for setting an exponent to 0

# %% Load in data and preprocess

# Read in incompressible PG data from Gonzalo
GonzaloDat = np.load(gonzalo_taudata)
npoints = GonzaloDat['X'].shape[0]
PiY = GonzaloDat['Y'] / GonzaloDat['X'][:,[0]]**2
# Pop out second to last column (kinetic energy metric \int_0^\delta U^2dy I'm not bothering with)
X = np.delete(GonzaloDat['X'],-2,axis=1)
# Add rhoe=rhow=1, muw=mue
X = np.hstack((X, [1, 1]*np.ones((npoints,2)), X[:,4][:,np.newaxis]))

# Build data object
# dat = tau_data(X, PiY)
# Try just compressible data to see what happens:
dat = tau_data()
ivars = [dat._vars_all.index(x) for x in Vars]

# Get Wenzel data
cases = []
for cs in glob(f"{fpaths.Wenzel2019_path}/*.dill"):
    c = db.load_case(cs)

    # Extract edge values based on chosen criterion
    nx, ny = c.u.shape
    c.delta99, ide = c.find_edge('delta99')
    ide = np.transpose(np.array(list(zip(range(nx),ide))))
    c.ue = c.u[*ide]
    c.mue = c.mu[*ide]
    c.rhoe = c.rho[*ide]
    c.ae = c.a[*ide]

    cases.append(c)
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
    _ = ax.set_xticks(range(len(Vars)), [f"${v}$" for v in Varlbls])
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_exp.pdf')

    # Plot Pi groups against each other
    X, PiY, _ = dat.get_data(Vars)
    # Ni = 1
    PiX = np.array([IT_Pi.getPiIfromXe(X, e_.squeeze()) for e_ in e[0]])
    # Me = X[:,0]/X[:,-1]
    kmin = MI[0].argmin()
    print("Optimal exponents, Ni=1:")
    IT_Pi.pretty_exps(e[0][kmin], Varlbls, prnt=True)
    fig, ax = plt.subplots(figsize=(6,6), layout='constrained')
    # Cmin, Cmax = np.min(Me), np.max(Me)
    # cmap = cm.ScalarMappable(colors.Normalize(vmin=Cmin, vmax=Cmax),"rainbow")
    ax.scatter(PiX[kmin], PiY, c='k', s=2, alpha=0.75)
    # ax.scatter(PiX[kmin], PiY, c=Me, cmap='rainbow', s=2, alpha=0.75)
    # cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(r"$M_e$",labelpad=15)
    ax.set_xlabel(r"$\Pi_1$")
    ax.set_ylabel(r"$\Pi_\tau$", rotation=0)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni1_space.pdf')
    
    # Ni = 2
    PiX2 = np.array([[IT_Pi.getPiIfromXe(X, e_) for e_ in ek] for ek in e[1]])
    kmin = MI[1].argmin()
    print("Optimal exponents, Ni=2:")
    IT_Pi.pretty_exps(e[1][kmin], Varlbls, prnt=True)
    Cmin, Cmax = np.min(PiY), np.max(PiY)
    cmap = cm.ScalarMappable(colors.Normalize(vmin=Cmin*1e3, vmax=Cmax*1e3),"rainbow")
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    ax.scatter(PiX2[kmin,0], PiX2[kmin,1], c=PiY*1e3, cmap='rainbow')  # , edgecolors='k')
    cbar = fig.colorbar(cmap,ax=ax); cbar.set_label(r"$\Pi_\tau\times10^3$",labelpad=15)
    ax.set_xlabel(r"$\Pi_1$")
    ax.set_ylabel(r"$\Pi_2$", rotation=0)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni2_space.pdf')

# %%
