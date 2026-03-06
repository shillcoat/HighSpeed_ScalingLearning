# Imports and plot formatting
import numpy as np
import IT_Pi
from glob import glob
import tomllib
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'text.usetex': True, 
                 'font.size': 14, 
                 'font.family': 'serif'})

from database import db_config as db
import database.filepaths as fpaths
gonzalo_taudata = fpaths.db_path + "/../GonzaloData/tau_data.npz"

from ITPi_classes import tau_data
from ITPi_plotting import *

# PiY = tau_w/(rho*U_e^2)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run IT_Pi on tau data")
    parser.add_argument("-c", "--config", type=str, default="inputfiles/ITPi_tau.toml", help="Path to toml config file")
    parser.add_argument("-p", "--plot", action='store_true', 
                        help="Flag to only run plotting routines (assumes IT_Pi has already been run and output files are in place)")
    args = parser.parse_args()

    # Script settings/inputs: read from toml config file
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    Nk = config['N_training'] # Number of times to repeat the training
    Njobs = config['N_jobs'] # Number of parallel jobs (-1 for all cores)
    Ntrain = config['N_points'] # Number of data points to use for training
    NPi = config['N_Pi'] # List of number of Pi groups to consider
    Vars = config['Vars'] # Variables to consider
    Varlbls = config['Varlbls'] # LaTeX variable labels for plotting, in same order as Vars
    output_path = config['output_path']
    exp_thresh = config['exp_threshold'] # Threshold for setting exponent to zero
    train_ratio = config['train_ratio']  # Ratio of data to use for training
    bl_edge = config['bl_edgetype'] # Criterion for identifying boundary layer edge (e.g. 'delta99', 'delta1k')
    datalst = config['data'] # Data to include in training (e.g. ['Gonzalo', 'Wenzel', 'Volpiani'])

    if not os.path.exists(output_path):
        os.mkdir(output_path)

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
        dat = tau_data(X, PiY, ID=np.array(['Gonzalo']*X.shape[0],dtype='T').reshape(-1,1))
    else: dat = tau_data() # Just initialize empty data object and extract from cases

    cases = []
    IDs = []

    # Wenzel data
    if 'Wenzel' in datalst:
        for cs in glob(f"{fpaths.Wenzel2019_path}/*.dill"):
            c = db.load_case(cs)
            cname = cs.split('/')[-1][:-5]

            # Extract edge values based on chosen criterion
            nx, ny = c.u.shape
            c.delta99, ide = c.find_edge(bl_edge)
            ide = np.transpose(np.array(list(zip(range(nx),ide))))
            c.ue = c.u[*ide]
            c.mue = c.mu[*ide]
            c.rhoe = c.rho[*ide]

            cases.append(c)
            IDs.append(f'Wenzel.{cname}')

    # Volpiani data
    if 'Volpiani' in datalst:
        for cs in glob(f"{fpaths.Volpiani2020_path}/*.dill"):
            c = db.load_case(cs)
            cname = cs.split('/')[-1][:-5]

            # Extract edge values based on chosen criterion
            nx, ny = c.u.shape
            _, ide = c.find_edge(bl_edge, sigma_smooth=[8,2])
            ide = np.transpose(np.array(list(zip(range(nx),ide))))
            c.ue = c.u[*ide]
            c.mue = c.mu[*ide]
            c.rhoe = c.rho[*ide]
            c.muw = c.muw * np.ones_like(c.x)

            cases.append(c)
            IDs.append(f'Volpiani.{cname}')

    # Larsson data: I'm a little unsure about including this bc reconstruction was severe
    # Also may overwhelm Wenzel data (only compressible data with actual PG)
    if 'Larsson' in datalst:
        for cs in glob(f"{fpaths.LarssonGroupBL_path}/*.dill"):
            c = db.load_case(cs)
            cname = cs.split('/')[-1][:-5]

            # Data is very limited for this dataset: use what we have
            c.ue = np.ones_like(c.x)
            c.mue = c.muinf * np.ones_like(c.x)
            c.rhoe = c.rhoinf
            c.muw = c.muw * np.ones_like(c.x)
            c.P = np.zeros_like(c.x) # Just give it something random so assertion doesn't throw error. Isn't used

            cases.append(c)
            IDs.append(f'Larsson.{cname}')

    # _ = dat.extract_vars(cases, grad_smooth=[5,1], IDs=IDs) # Pretty agressive edge artifacts, don't think necessary
    _ = dat.extract_vars(cases, IDs=IDs)
    del cases # Free up memory once data is extracted

    # Split data between training and validation sets (save these to go back to)
    traindat, validdat = dat.split_train_valid(train_ratio)
    np.savez(f'{output_path}/tau_ITPI_{"_".join(Vars)}_train_valid.npz', 
             traindat_X=traindat._X, traindat_Y=traindat._Y, validdat_X=validdat._X, validdat_Y=validdat._Y)

    if not args.plot:
        # Run IT_Pi
        for k in range(Nk):
            print(f'Iteration = {k+1}')
            for num_input in NPi:
                X, Y, _, basis_matrices = traindat.get_data(Vars, Npts=Ntrain)

                results = IT_Pi.run(
                    X, Y,
                    basis_matrices,
                    num_input=num_input,
                    estimator="kraskov",
                    estimator_params={"k": 5},
                    n_jobs=Njobs,
                    optimize_output=False,
                )

                results = results['orig_results']  # For compatibility with original IT-Pi function

                fnm = f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{num_input}_{k}_output'
                np.savez(fnm, **results)

    # Plot some results
    # Visualize exponents
    e, MI = [], []
    fig, axs = plt.subplots(1, 2, figsize=(12,5), sharex=True, layout='constrained')
    for Ni in NPi:
        flist = glob(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_*_output.npz')
        _, _, e_, MI_ = plt_exps(flist, Ni, tau_data, Vars, Varlbls, ax=axs[Ni-1], exp_thresh=exp_thresh)
        e.append(e_)
        MI.append(MI_)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_exp.pdf')

    # Print optimal exponents
    for i, Ni in enumerate(NPi):
        kmin = MI[i].argmin()
        print(f"Optimal exponents, Ni={Ni}:")
        IT_Pi.pretty_exps(e[i][kmin], Varlbls, prnt=True)

    # Plot Pi groups against each other
    X, PiY, IDs, _ = dat.get_vars(Vars)

    if 1 in NPi:
        # Ni = 1
        fig, ax = plt.subplots(figsize=(6,6), layout='constrained')
        # Color points by dataset (in same order as datalst)
        cols = np.zeros_like(IDs, dtype=int)
        if len(datalst) > 1:
            for i in range(len(datalst)):
                cols += (i+1)*(np.char.find(IDs,datalst[i])>=0).astype(int)
        else:
            ids = np.unique(IDs)
            for i, name in enumerate(ids):
                cols += (i+1)*(IDs == name).astype(int)
            assert np.max(cols) <= len(ids)  # Basic sanity check that color mapping makes sense/didn't double count anything
        print(f"Dataset color mapping: {datalst if len(datalst)>1 else ids}")
        kmin = MI[0].argmin()
        plt_1Pi(X, PiY, e[0][kmin], Vars, Varlbls, PiYlbl=r"$\Pi_\tau$", ax=ax, s=2, alpha=0.75, colQ=cols, colLbl='Dataset')
        fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni1_space.pdf')

    if 2 in NPi:    
        # Ni = 2
        fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
        kmin = MI[1].argmin()
        plt_2Pi(X, PiY, e[1][kmin], Vars, Varlbls, PiYlbl=r"$\Pi_\tau$", ax=ax)
        fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni2_space.pdf')

# %%
