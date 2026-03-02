# Imports and plot formatting
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
from ITPi_plotting import *

if __name__ == "__main__":
    # Script settings/inputs: read from toml config file
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
    datalst = config['data'] # Data to include in training (e.g. ['Gonzalo', 'Wenzel', 'Volpiani'])

    # Read in incompressible PG data from Gonzalo
    GonzaloDat = np.load(gonzalo_taudata)
    npoints = GonzaloDat['X'].shape[0]
    PiY = GonzaloDat['Y'] / GonzaloDat['X'][:,[0]]**2
    # Pop out second to last column (kinetic energy metric \int_0^\delta U^2dy I'm not bothering with)
    X = np.delete(GonzaloDat['X'],-2,axis=1)
    # Add rhoe=rhow=1, muw=mue
    X = np.hstack((X, [1, 1]*np.ones((npoints,2)), X[:,4][:,np.newaxis]))

    ivars = [tau_data._vars_all.index(x) for x in Vars]

    if 'Gonzalo' in datalst:
        dat = tau_data(X, PiY)
    else: dat = tau_data() # Just initialize empty data object and extract from cases

    cases = []

    # Wenzel data
    if 'Wenzel' in datalst:
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
    if 'Volpiani' in datalst:
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
    if 'Larsson' in datalst:
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
    del cases # Free up memory once data is extracted

    # Split data between training and validation sets (save these to go back to)
    traindat, validdat = dat.split_train_valid(train_ratio)
    np.savez(f'{output_path}/tau_ITPI_{"_".join(Vars)}_train_valid.npz', 
             traindat_X=traindat._X, traindat_Y=traindat._Y, validdat_X=validdat._X, validdat_Y=validdat._Y)

    # Run IT_Pi
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
        flist = glob(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_*_output.npz')
        _, _, e_, MI_ = plt_exps(flist, Ni, Vars, Varlbls, ax=axs[Ni-1], exp_thresh=exp_thresh)
        e.append(e_)
        MI.append(MI_)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_exp.pdf')

    # Print optimal exponents
    for i, Ni in enumerate([1,2]):
        kmin = MI[i].argmin()
        print(f"Optimal exponents, Ni={Ni}:")
        IT_Pi.pretty_exps(e[i][kmin], Varlbls, prnt=True)

    # Plot Pi groups against each other
    X, PiY, _ = dat.get_vars(Vars)
    # Ni = 1
    fig, ax = plt.subplots(figsize=(6,6), layout='constrained')
    kmin = MI[0].argmin()
    plt_1Pi(X, PiY, e[0][kmin], Vars, Varlbls, PiYlbl=r"$\Pi_\tau$", ax=ax, s=2, alpha=0.75)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni1_space.pdf')
    
    # Ni = 2
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    kmin = MI[1].argmin()
    plt_2Pi(X, PiY, e[1][kmin], Vars, Varlbls, PiYlbl=r"$\Pi_\tau$", ax=ax)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni2_space.pdf')

# %%
