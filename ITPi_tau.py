# Imports and plot formatting
import numpy as np
import IT_Pi
from glob import glob
import tomllib
import os
from argparse import ArgumentParser
import multiprocessing as mp

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

# PiY = tau_w/(rho_w*U_e^2)

def extract_cases(datalst, dat_object=None, verbose=True):
    if dat_object is None:
        dat_object = tau_data()
    
    cases = []
    IDs = []

    if 'Gonzalo' in datalst:
        # Read in incompressible PG data from Gonzalo
        GonzaloDat = np.load(gonzalo_taudata)
        npoints = GonzaloDat['X'].shape[0]
        PiY = GonzaloDat['Y'] / GonzaloDat['X'][:,[0]]**2
        # Pop out second to last column (kinetic energy metric \int_0^\delta U^2dy I'm not bothering with)
        X = np.delete(GonzaloDat['X'],-2,axis=1)
        # Add rhoe=rhow=1, muw=mue
        X = np.hstack((X, [1, 1]*np.ones((npoints,2)), X[:,4][:,np.newaxis]))
        dat_object.append_data(X, PiY, ID_new=np.array(['Gonzalo']*npoints,dtype='T').reshape(-1,1))
        if verbose: print(f"Total number of points from Gonzalo dataset: {npoints}")

    # Wenzel data
    if 'Wenzel' in datalst:
        npts = 0
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
            npts += len(c.x)
        if verbose: print(f"Total number of points from Wenzel dataset: {npts}")

    # Volpiani data
    if 'Volpiani' in datalst:
        npts = 0
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
            npts += len(c.x)
        if verbose: print(f"Total number of points from Volpiani dataset: {npts}")
    
    # Larsson data: I'm a little unsure about including this bc reconstruction was severe
    # Also may overwhelm Wenzel data (only compressible data with actual PG)
    if 'Larsson' in datalst:
        npts = 0
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
            npts += len(c.x)
        if verbose: print(f"Total number of points from Larsson dataset: {npts}")

    _ = dat_object.extract_vars(cases, IDs=IDs)
    return dat_object

##################################################################################################################

if __name__ == "__main__":
    # Configure multiprocessing so I can keep editing things in the background without breaking running scripts
    mp.set_start_method('forkserver')

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
    Ylbl = config['Ylbl']  # LaTeX label for target variable
    output_path = config['output_path']
    exp_thresh = config['exp_threshold'] # Threshold for setting exponent to zero
    train_ratio = config['train_ratio']  # Ratio of data to use for training
    bl_edge = config['bl_edgetype'] # Criterion for identifying boundary layer edge (e.g. 'delta99', 'delta1k')
    datalst = config['data'] # Data to include in training (e.g. ['Gonzalo', 'Wenzel', 'Volpiani'])
    oo = config['optimize_output']  # Whether to optimize the output group as well
    estimator = config['estimator']  # Estimator to use for mutual information ("kraskov" or "binning")
    estimator_params = config['estimator_params']  # Parameters for the chosen estimator

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ivars = [tau_data._vars_all.index(x) for x in Vars]

    dat = extract_cases(datalst)
    print(f"Total number of data points after extraction: {len(dat._X)}")

    # Split data between training and validation sets (save these to go back to)
    traindat, validdat = dat.split_train_valid(train_ratio)
    np.savez(f'{output_path}/tau_ITPI_{"_".join(Vars)}_train_valid.npz', 
             traindat_X=traindat._X, traindat_Y=traindat._Y, validdat_X=validdat._X, validdat_Y=validdat._Y)

    if not args.plot:
        # Run IT_Pi
        for k in range(Nk):
            print(f'\nIteration = {k+1}')
            print('-'*60)
            for num_input in NPi:
                X, Y, _, basis_matrices = traindat.get_data(Vars, Npts=Ntrain)

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
                    iy = Vars.index('y')
                    IT_Pi.rescale_output(results, basis_matrices, iy)
                    print(f"a_list_o: {results['a_list_o']}")
                    print(f"Output coefs: {results['output_coef']}")

                fnm = f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{num_input}_{k}_output'
                np.savez(fnm, **results)

    # Plot some results
    # Visualize exponents
    e_in, e_out, MI = [], [], []
    fig, axs = plt.subplots(1, 2, figsize=(12,5), sharex=True, layout='constrained')
    for Ni in NPi:
        flist = glob(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_*_output.npz')
        print(f"Found {len(flist)} files for Ni={Ni}")
        _, _, ein_, eout_, MI_ = plt_exps(flist, Ni, tau_data, Vars, Varlbls, ax=axs[Ni-1], exp_thresh=exp_thresh,
                                          inorm=None)
        e_in.append(ein_)
        e_out.append(eout_)
        MI.append(MI_)
    fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni{Ni}_exp.pdf')

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
        plt_1Pi(X, Y, ei, eo, PiYlbl=r"$\Pi_\tau$", ax=ax, s=2, alpha=0.75, colQ=cols, colLbl='Dataset')
        fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni1_space.pdf')

    if 2 in NPi:    
        # Ni = 2
        fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
        kmin = MI[1].argmin()
        ei = e_in[1][kmin]
        eo = e_out[1][kmin] if oo else None
        plt_2Pi(X, Y, ei, eo, PiYlbl=r"$\Pi_\tau$", ax=ax)
        fig.savefig(f'{output_path}/tau_ITPI_{"_".join(Vars)}_Ni2_space.pdf')

# %%
