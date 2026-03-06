# Imports and plot formatting
import numpy as np
import scipy
import IT_Pi
from IT_Pi.funcs import norme
from numpy.linalg import matrix_rank

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'text.usetex': True, 
                 'font.size': 14, 
                 'font.family': 'serif'})
import database.filepaths as fpaths

from ITPi_classes import Yuan_U_data

# PiY = u/utau? Or get ITPi to optimize it?

if __name__ == "__main__":
    # Vars = ["y", "rho", "mu", "rhow", "muw", "utau"]
    # dat = Yuan_U_data()

    # mat_data = scipy.io.loadmat(fpaths.Yuan_path)  # Velocity data from Yuan's IT-Pi example
    # dat.append_data(mat_data['input'], mat_data['output'], ID_new='Yuan')

    # # Run IT_Pi
    # X, Y, _, basis_matrices = dat.get_data(Vars, Npts=2000)

    mat_data  = scipy.io.loadmat(fpaths.Yuan_path)
    X         = mat_data['input'] # Load the .mat file, [Y1,Rho1,Mu1,Rhow,Muw,Utau]
    Y         = mat_data['output'] # The output is u/utau
    print('input shape: ', X.shape)
    D_in      = np.matrix('1 -3 -1 -3 -1  1; ' \
                        '0  0 -1  0 -1 -1; ' \
                        '0  1  1  1  1  0')
    num_input = 1

    print("Rank of D_in:", matrix_rank(D_in))
    print("D_in matrix:\n", D_in)
    num_basis           = D_in.shape[1] - matrix_rank(D_in)
    basis_matrices      = IT_Pi.calc_basis(D_in, num_basis)
    basis_matrices[2,:] = -basis_matrices[2,:]  # Adjust the third basis vector

    results = IT_Pi.run(
        X, Y,
        basis_matrices,
        num_input=1,
        # estimator="kraskov",
        # estimator_params={"k": 5},
        estimator="binning",
        estimator_params={"num_bins": 40},
        seed=666666,
        n_jobs=1,
        optimize_output=True,
    )

    raw_results = results["raw_results"]

    # From Yuan's script
    input_PI  = results["input_PI"]
    output_PI = results["output_PI"]

    # These don't seem right?
    epsilon   = results["irreducible_error"]
    uq        = results["uncertainty"]

    mask = np.asarray(input_PI != 0).flatten()
    epsilon,uq = IT_Pi.calculate_bound_and_uq(np.log(np.asarray(input_PI)[mask].reshape(-1,1)), np.asarray(output_PI)[mask], num_trials=1)
    print('Irreducible error: ', epsilon)
    print('Uncertainty: ', uq)
    a_list    = results["input_coef_basis"]
    a_list_o  =results["output_coef_basis"]
    variables_pi_i = ['(\\frac{\\rho}{\\rho_w})', '(\\frac{\\mu}{\\mu_w})', '(\\frac{y \\rho u_{\\tau}}{\\mu})']
    variables_pi_o = ['(\\frac{\\rho}{\\rho_w})', '(\\frac{\\mu}{\\mu_w})', '(\\frac{y \\rho u_{\\tau}}{\\mu})']
    Pi_i_lab = IT_Pi.create_labels(a_list, variables_pi_i)
    Pi_o_lab = IT_Pi.create_labels(a_list_o, variables_pi_o)
    # Print the labels
    for j, label in enumerate(Pi_i_lab):
        print(f'Pi_i_lab[{j}] = {label}')
    for j, label in enumerate(Pi_o_lab):
        Pi_o_lab[j] = r'$\frac{u}{u_{\tau}} \cdot ' + label[1:]  # Remove the first '$' to merge the LaTeX strings properly
    for j, label in enumerate(Pi_o_lab):
        print(f'Pi_o_lab[{j}] = {label}')