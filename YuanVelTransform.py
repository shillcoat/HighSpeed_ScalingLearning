# Imports and plot formatting
import numpy as np
import scipy
import IT_Pi
from ITPi_plotting import *
from IT_Pi.funcs import norme
from numpy.linalg import matrix_rank
import pdb

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'text.usetex': True, 
                 'font.size': 14, 
                 'font.family': 'serif'})
import database.filepaths as fpaths

from ITPi_classes import Yuan_U_data

if __name__ == "__main__":
    Vars = ["y", "rho", "mu", "rhow", "muw", "utau"]
    dat = Yuan_U_data()

    mat_data = scipy.io.loadmat(fpaths.Yuan_path)  # Velocity data from Yuan's IT-Pi example
    dat.append_data(mat_data['input'], mat_data['output'], ID_new=mat_data['index'])

    # Run IT_Pi
    X, Y, IDs, basis_matrices = dat.get_data(Vars, Npts=2000)

    results = IT_Pi.run(
        X, Y,
        basis_matrices,
        num_input=1,
        # estimator="kraskov",
        # estimator_params={"k": 5},
        estimator="binning",
        estimator_params={"num_bins": 40},
        seed=666666,
        n_jobs=10,
        optimize_output=True,
    )

    # Remove contribution of input Pi group to cancel out output group y dependence (non-physical)
    iy = Vars.index('y')
    ib, _ = np.argwhere(basis_matrices[:,0]!=0)[0] # Basis vector with y dependence
    for a_o in results['a_list_o']:
        ratio = a_o[ib] / results['a_list_i'][0][ib] if results['a_list_i'][0][ib] != 0 else 1.0
        a_o -= results['a_list_i'][0] * ratio
    results['output_coef'] = np.array(np.dot(results['a_list_o'], basis_matrices))

    print(f"a_list_o: {results['a_list_o']}")
    print(f"Output coefs: {results['output_coef']}")

    # Post-processing and plotting
    _, _, e_in, e_out, MI = plt_exps([results],1,dat,Vars)
    plt.show()

    fig, ax = plt.subplots(figsize=(6,6), layout='constrained')
    cols = np.zeros_like(IDs, dtype=int)
    ids = np.unique(IDs)
    for i, name in enumerate(ids):
        cols += (i+1)*(IDs == name).astype(int)
    assert np.max(cols) <= len(ids)  # Basic sanity check that color mapping makes sense/didn't double count anything
    print(f"Dataset color mapping: {ids}")
    kmin = MI[0].argmin()
    ei = e_in[0][kmin]
    eo = e_out[0][kmin]
    plt_1Pi(X, Y, ei, eo, PiYlbl=r"$\Pi_U$", ax=ax, s=2, alpha=0.75, colQ=cols, colLbl='Dataset')
    ax.set_xscale('log')
    plt.show()

    pdb.set_trace()

    # # From Yuan's script
    # input_PI  = results["input_PI"]
    # output_PI = results["output_PI"]
    # epsilon   = results["irreducible_error"]
    # uq        = results["uncertainty"]

    # # Two differences from IT_Pi output: takes log of input_PI, and only considers nonzero input_PI values (since log(0) is undefined)
    # # Why take the log? Is it more physically meaningful this way? Need to understand what this actually is
    # mask = np.asarray(input_PI != 0).flatten()
    # epsilon,uq = IT_Pi.calculate_bound_and_uq(np.log(np.asarray(input_PI)[mask].reshape(-1,1)), np.asarray(output_PI)[mask], num_trials=1)
    # print('Irreducible error: ', epsilon)
    # print('Uncertainty: ', uq)
    # a_list    = results["input_coef_basis"]
    # a_list_o  =results["output_coef_basis"]
    # variables_pi_i = ['(\\frac{\\rho}{\\rho_w})', '(\\frac{\\mu}{\\mu_w})', '(\\frac{y \\rho u_{\\tau}}{\\mu})']
    # variables_pi_o = ['(\\frac{\\rho}{\\rho_w})', '(\\frac{\\mu}{\\mu_w})', '(\\frac{y \\rho u_{\\tau}}{\\mu})']
    # Pi_i_lab = IT_Pi.create_labels(a_list, variables_pi_i)
    # Pi_o_lab = IT_Pi.create_labels(a_list_o, variables_pi_o)
    # # Print the labels
    # for j, label in enumerate(Pi_i_lab):
    #     print(f'Pi_i_lab[{j}] = {label}')
    # for j, label in enumerate(Pi_o_lab):
    #     Pi_o_lab[j] = r'$\frac{u}{u_{\tau}} \cdot ' + label[1:]  # Remove the first '$' to merge the LaTeX strings properly
    # for j, label in enumerate(Pi_o_lab):
    #     print(f'Pi_o_lab[{j}] = {label}')