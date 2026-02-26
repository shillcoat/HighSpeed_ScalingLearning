# %% Imports and plot formatting
import numpy as np
import IT_Pi
from glob import glob
import tomllib

import matplotlib.pyplot as plt
from matplotlib import cm, colors, rcParams
rcParams.update({'text.usetex': True, 'font.size': 14, 'font.family': 'serif'})

from database import db_config as db
import database.filepaths as fpaths
gonzalo_taudata = fpaths.db_path + "/../GonzaloData/tau_data.npz"

from ITPi_classes import tau_data

# %% Script settings/inputs: read from toml config file
with open("inputfiles/ITPi_tau.toml", "rb") as f:
    config = tomllib.load(f)
Nk = config['N_training'] # Number of times to repeat the training
Njobs = config['N_jobs'] # Number of parallel jobs (-1 for all cores)
Ntrain = config['N_points'] # Number of data points to use for training
Vars = config['Vars'] # Variables to consider
Varlbls = config['Varlbls'] # LaTeX variable labels for plotting, in same order as Vars
output_path = config['output_path']
exp_thresh = config['exp_threshold'] # Threshold for setting exponent to zero
train_ratio = config['train_ratio']  # Ratio of data to use for training
bl_edge = config['bl_edgetype'] # Criterion for identifying boundary layer edge (e.g. 'delta99', 'delta1k')

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
dat = tau_data(X, PiY)
# Try just compressible data to see what happens:
# dat = tau_data()
ivars = [dat._vars_all.index(x) for x in Vars]

cases = []

# Wenzel data
for cs in glob(f"{fpaths.Wenzel2019_path}/*.dill"):
    c = db.load_case(cs)

    # Extract edge values based on chosen criterion
    nx, ny = c.u.shape
    c.delta99, ide = c.find_edge(bl_edge)
    ide = np.transpose(np.array(list(zip(range(nx),ide))))
    c.ue = c.u[*ide]
    c.mue = c.mu[*ide]
    c.rhoe = c.rho[*ide]

    cases.append(c)

# Volpiani data
for cs in glob(f"{fpaths.Volpiani2020_path}/*.dill"):
    c = db.load_case(cs)

    # Extract edge values based on chosen criterion
    nx, ny = c.u.shape
    c.delta99, ide = c.find_edge(bl_edge)
    ide = np.transpose(np.array(list(zip(range(nx),ide))))
    c.ue = c.u[*ide]
    c.mue = c.mu[*ide]
    c.rhoe = c.rho[*ide]
    c.muw = c.muw * np.ones_like(c.x)

    cases.append(c)

# Larsson data: I'm a little unsure about including this bc reconstruction was severe
# Also may overwhelm Wenzel data (only compressible data with actual PG)
for cs in glob(f"{fpaths.LarssonGroupBL_path}/*.dill"):
    c = db.load_case(cs)

    # Data is very limited for this dataset: use what we have
    c.ue = np.ones_like(c.x)
    c.mue = c.muinf * np.ones_like(c.x)
    c.rhoe = c.rhoinf
    c.muw = c.muw * np.ones_like(c.x)
    c.P = np.zeros_like(c.x) # Just give it something random so assertion doesn't throw error. Isn't used

    cases.append(c)

_ = dat.extract_vars(cases)

# Split data between training and validation sets
traindat, validdat = dat.split_train_valid(train_ratio)

# %% Run IT_Pi
if __name__ == "__main__":  # Necessary for parallelization
    for k in range(Nk):
        print(f'Iteration = {k+1}')
        for num_input in (1, 2):
            X, Y, basis_matrices = traindat.get_data(Vars, Npts=Ntrain)

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
    X, PiY, _ = dat.get_vars(Vars)
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
