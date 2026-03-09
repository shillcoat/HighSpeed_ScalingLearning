# Imports and plot formatting
import numpy as np
import scipy
import IT_Pi
from glob import glob
import tomllib
import os
from argparse import ArgumentParser
import multiprocessing

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'text.usetex': True, 
                 'font.size': 14, 
                 'font.family': 'serif'})

from database import db_config as db
import database.filepaths as fpaths

from ITPi_classes import U_data
from ITPi_plotting import *

import pdb

# Y = u/utau, can allow IT-Pi to optimize it

def extract_cases(caselst, dat_object=None, remove_wake=True, resample=True):
    if dat_object is None:
        dat_object = U_data()
    
    cases = []
    IDs = []

    # Handle Yuan's data a little differently
    # Only supported when the variables of interest are ['y', 'rho', 'mu', 'rhow', 'muw', 'utau']
    # or some subset of those
    if 'Yuan' in caselst:
        mat_data = scipy.io.loadmat(fpaths.Yuan_path)  # Velocity data from Yuan's IT-Pi example
        X, Y, ids = mat_data['input'], mat_data['output'], mat_data['index']
        N = X.shape[0]
        X = np.hstack([np.zeros((N,2)), X[:,[5,0,2,4,1,3]], np.zeros((N,1))])
        yuan_ids = np.array([f'Yuan.{id[0]}' for id in ids], dtype='T').reshape(-1,1)
        dat_object.append_data(X, Y, ID_new=yuan_ids)
    
    # Wenzel data
    if 'Wenzel' in caselst:
        for cs in glob(f"{fpaths.Wenzel2019_path}/*.dill"):
            if 'ZPG' not in cs: # Only use ZPG data for now
                continue
            c = db.load_case(cs)
            cname = cs.split('/')[-1][:-5]
            c.ue = c.uinf

            cases.append(c)
            IDs.append(f'Wenzel.{cname}')

    # Volpiani data
    if 'Volpiani' in caselst:
        npts = 0
        for cs in glob(f"{fpaths.Volpiani2020_path}/*.dill"):
            cname = cs.split('/')[-1][:-5]
            # if 'twtr100' not in cname:
            #     continue # Skip all diabatic cases for the minute
            
            c = db.load_case(cs)
            c.muw = c.muw * np.ones_like(c.x)
            c.ue = c.uinf * np.ones_like(c.x)

            cases.append(c)
            IDs.append(f'Volpiani.{cname}')

            npts += len(c.y)*len(c.x)
        print(f"Total number of data points from Volpiani: {npts}")

    # Zhang data
    if 'Zhang' in caselst:
        npts = 0
        for cs in glob(f"{fpaths.Zhang2018_path}/*.dill"):
            cname = cs.split('/')[-1][:-5]
            # if 'Tw' in cname:
            #     continue # Skip all diabatic cases for the minute

            c = db.load_case(cs)
            c.ue = c.uinf

            cases.append(c)
            IDs.append(f'Zhang.{cname}')

            npts += len(c.y)
        print(f"Total number of data points from Zhang: {npts}")

    # Trettel data
    if 'Trettel' in caselst:
        npts = 0
        for cs in glob(f"{fpaths.Trettel2016_path}/*.dill"):
            cname = cs.split('/')[-1][:-5]
            c = db.load_case(cs)

            cases.append(c)
            IDs.append(f'Trettel.{cname}')

            npts += len(c.y)
        print(f"Total number of data points from Trettel: {npts}")

    # Modesti data
    if 'Modesti' in caselst:
        npts = 0
        for cs in glob(f"{fpaths.Modesti2016_path}/*.dill"):
            cname = cs.split('/')[-1][:-5]
            c = db.load_case(cs)
            
            c.ue = c.u[...,-1]
            c.P = np.zeros_like(c.x) # Just give it something random so assertion doesn't throw error. Isn't used

            cases.append(c)
            IDs.append(f'Modesti.{cname}')

            npts += len(c.y)
        print(f"Total number of data points from Modesti: {npts}")

    # Sillero data (incompressible)
    if 'Sillero' in caselst:
        npts = 0
        for cs in glob(f"{fpaths.Sillero2014_path}/*.dill"):
            cname = cs.split('/')[-1][:-5]
            c = db.load_case(cs)
            c.x = 0
            c.rho = np.ones_like(c.y); c.rhow = 1.0
            c.mu = c.nu * c.rho; c.muw = c.nu.squeeze()
            c.ue = c.uinf

            c.P = np.zeros_like(c.x) # Just give it something random so assertion doesn't throw error. Isn't used

            cases.append(c)
            IDs.append(f'Sillero.{cname}')

            npts += len(c.y)
        print(f"Total number of data points from Sillero: {npts}")

    # Larsson data: I'm a little unsure about including this bc reconstruction was severe
    if 'Larsson' in caselst:
        for cs in glob(f"{fpaths.LarssonGroupBL_path}/*.dill"):
            c = db.load_case(cs)
            cname = cs.split('/')[-1][:-5]

            # Data is very limited for this dataset: use what we have
            c.ue = np.ones_like(c.x)
            c.muw = c.muw * np.ones_like(c.x)
            c.P = np.zeros_like(c.x) # Just give it something random so assertion doesn't throw error. Isn't used

            cases.append(c)
            IDs.append(f'Larsson.{cname}')

    # Load Sillero reference incompressible profile for later comparison
    c = db.load_case(f"{fpaths.Sillero2014_path}/Re_theta4500.dill")
    ypref = c.yplus
    upref = c.uplus

    _ = dat_object.extract_vars(cases, grad_smooth=[5,1], IDs=IDs, remove_wake=remove_wake, resample=resample)
    return dat_object, ypref, upref

##################################################################################################################

if __name__ == "__main__":
    # Configure multiprocessing so I can keep editing things in the background without breaking running scripts
    multiprocessing.set_start_method('forkserver')

    parser = ArgumentParser(description="Run IT_Pi on U data")
    parser.add_argument("-c", "--config", type=str, default="inputfiles/ITPi_U.toml", help="Path to toml config file")
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
    Ylbl = config['Ylbl']  # LaTeX label for target variable
    output_path = config['output_path']
    exp_thresh = config['exp_threshold'] # Threshold for setting exponent to zero
    train_ratio = config['train_ratio']  # Ratio of data to use for training
    datalst = config['data'] # Data to include in training (e.g. ['Gonzalo', 'Wenzel', 'Volpiani'])
    oo = config['optimize_output']  # Whether to optimize the output group as well
    rwake = config['remove_wake']  # Whether to remove wake data points (y > delta99)
    estimator = config['estimator']  # Estimator to use for mutual information ("kraskov" or "binning")
    estimator_params = config['estimator_params']  # Parameters for the chosen estimator
    resample = config['resample']  # Whether to resample data on log scale to give more near-wall points

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ivars = [U_data._vars_all.index(x) for x in Vars]
    dat, ypref, upref = extract_cases(datalst, remove_wake=rwake, resample=resample)
    print(f"Total number of data points after extraction: {len(dat._X)}")

    # Split data between training and validation sets (save these to go back to)
    traindat, validdat = dat.split_train_valid(train_ratio)
    np.savez(f'{output_path}/U_ITPI_{"_".join(Vars)}_train_valid.npz', 
             traindat_X=traindat._X, traindat_Y=traindat._Y, validdat_X=validdat._X, validdat_Y=validdat._Y)

    # Run IT_Pi
    if not args.plot:
        for k in range(Nk):
            print(f'Iteration = {k+1}')
            for num_input in NPi:
                X, Y, _, basis_matrices = traindat.get_data(Vars, Npts=Ntrain)

                print(f"basis_matrices:\n{basis_matrices}")

                results = IT_Pi.run(
                    X, Y,
                    basis_matrices,
                    num_input=num_input,
                    estimator=estimator,
                    estimator_params=estimator_params,
                    n_jobs=Njobs,
                    optimize_output=oo,
                )

                print(f"a_list_i: {results['a_list_i']}")
                print(f"Input coefs: {results['input_coef']}")
                if oo: 
                    # Remove contribution of input Pi group to cancel out output group y dependence (non-physical)
                    iy = Vars.index('y')
                    ib, _ = np.argwhere(basis_matrices[:,0]!=0)[0] # Basis vector with y dependence
                    for a_o in results['a_list_o']:
                        ratio = a_o[ib] / results['a_list_i'][0][ib] if results['a_list_i'][0][ib] != 0 else 1.0
                        a_o -= results['a_list_i'][0] * ratio
                    results['output_coef'] = np.array(np.dot(results['a_list_o'], basis_matrices))

                    print(f"a_list_o: {results['a_list_o']}")
                    print(f"Output coefs: {results['output_coef']}")

                fnm = f'{output_path}/U_ITPI_{"_".join(Vars)}_Ni{num_input}_{k}_output'
                np.savez(fnm, **results)

    # Plot some results
    # Visualize exponents
    e_in, e_out, MI = [], [], []
    fig, axs = plt.subplots(1, 2, figsize=(12,5), sharex=True, layout='constrained')
    for Ni in NPi:
        flist = glob(f'{output_path}/U_ITPI_{"_".join(Vars)}_Ni{Ni}_*_output.npz')
        print(f"Found {len(flist)} files for Ni={Ni}")
        _, _, ein_, eout_, MI_ = plt_exps(flist, Ni, U_data, Vars, Varlbls, ax=axs[Ni-1], 
                                          exp_thresh=exp_thresh, inorm=Vars.index('y'))
        e_in.append(ein_)
        e_out.append(eout_)
        MI.append(MI_)
    fig.savefig(f'{output_path}/U_ITPI_{"_".join(Vars)}_Ni{Ni}_exp.pdf')

    # Print optimal exponents
    for i, Ni in enumerate(NPi):
        kmin = MI[i].argmin()
        print(f"Optimal exponents, Ni={Ni}:")
        print("Input groups:")
        IT_Pi.pretty_exps(e_in[i][kmin], Varlbls, prnt=True)
        if oo:
            print("Output groups:")
            eo = e_out[i][kmin]
            IT_Pi.pretty_exps(np.concatenate([eo,[[1]]],axis=-1), Varlbls + [Ylbl], prnt=True)

    # Plot Pi groups against each other
    if resample:
        # Don't want to plot with resampled data
        dat, ypref, upref = extract_cases(datalst, remove_wake=rwake, resample=False)
    X, Y, IDs, _ = dat.get_vars(Vars)

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
        ei = e_in[0][kmin]
        eo = e_out[0][kmin] if oo else None
        plt_1Pi(X, Y, ei, eo, PiYlbl=r"$\Pi_U$", ax=ax, s=2, alpha=0.75, colQ=cols, colLbl='Dataset')
        ax.set_xscale('log')
        ax.semilogx(ypref,upref,'k--')
        fig.savefig(f'{output_path}/U_ITPI_{"_".join(Vars)}_Ni1_space.png')
    
    if 2 in NPi:
        # Ni = 2
        fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
        kmin = MI[1].argmin()
        ei = e_in[1][kmin]
        eo = e_out[1][kmin] if oo else None
        plt_2Pi(X, Y, ei, eo, PiYlbl=r"$\Pi_U$", ax=ax)
        fig.savefig(f'{output_path}/U_ITPI_{"_".join(Vars)}_Ni2_space.png')

# %%
