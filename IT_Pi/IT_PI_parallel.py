import numpy as np
import random
from cma import CMAEvolutionStrategy
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from .IT_PI import MI_input_output, calc_pi, calc_pi_omega, calculate_bound_and_uq

###############################################
# Wrapper for parallel evaluation
###############################################
def evaluate_solution(params, num_para_input, num_para_output, basis_matrices, X, Y, 
                     num_basis, num_input, estimator, estimator_params, optimize_output):
    """Wrapper function for parallel evaluation of a single solution"""
    input_params = params[:num_para_input]
    output_params = params[num_para_input:] if optimize_output else None
    return MI_input_output(
        input_params,
        output_params,
        basis_matrices,
        X,
        Y,
        num_basis,
        num_input,
        estimator=estimator,
        estimator_params=estimator_params,
        optimize_output=optimize_output
    )


###############################################
# Main function with parallelization
###############################################
def main(
    X,
    Y,
    basis_matrices: np.matrix,
    num_input=1,
    popsize=300,
    maxiter=50000,
    num_trials=10,
    estimator="binning",
    estimator_params=None,
    seed=None,
    optimize_output=False,
    n_jobs=-1,  # New parameter for parallelization
    verbose=True
):
    """
    Main function with parallelization support.
    
    Parameters:
    -----------
    n_jobs : int, default=-1
        Number of parallel processes to use. 
        -1 means using all available CPUs.
        1 means no parallelization (sequential execution).
    verbose : bool, default=True
        Whether to print timing information.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Determine number of processes
    if n_jobs == -1:
        n_processes = cpu_count()
    elif n_jobs > 0:
        n_processes = min(n_jobs, cpu_count())
    else:
        n_processes = 1
    
    if verbose:
        print(f"Using {n_processes} process(es) for parallel evaluation")
    
    num_basis = basis_matrices.shape[0]
    print('-'*60)

    num_para_input = num_basis * num_input
    num_para_output = num_basis * optimize_output
    num_para_total = num_para_input + num_para_output
    print('num of parameters:', num_para_total)

    bounds = [[-2] * num_para_total, [2] * num_para_total]
    options = {
        'bounds': bounds,
        'maxiter': maxiter,
        'tolx': 1e-8,
        'tolfun': 1e-8,
        'popsize': popsize,
        'seed': seed if seed is not None else random.randint(0, 10000),
    }

    if estimator_params is None:
        estimator_params = {"num_bins": 50} if estimator == "binning" else {"k": 5}
    print(f"\nUsing estimator: '{estimator}' with hyperparameters: {estimator_params}\n")

    # Create partial function with fixed arguments for parallel evaluation
    eval_func = partial(
        evaluate_solution,
        num_para_input=num_para_input,
        num_para_output=num_para_output,
        basis_matrices=basis_matrices,
        X=X,
        Y=Y,
        num_basis=num_basis,
        num_input=num_input,
        estimator=estimator,
        estimator_params=estimator_params,
        optimize_output=optimize_output
    )

    # Initialize CMA-ES
    es = CMAEvolutionStrategy([0.1]*num_para_total, 0.5, options)
    
    # Track timing if verbose
    generation = 0
    start_time = np.nan
    if verbose: start_time = time.time()
    
    # Main optimization loop with parallelization
    if n_processes > 1:
        with Pool(processes=n_processes) as pool:
            while not es.stop():
                solutions = es.ask()
                
                # Parallel evaluation
                fitness_values = pool.map(eval_func, solutions)
                
                es.tell(solutions, fitness_values)
                es.disp()
                
                if verbose and generation % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Generation {generation}, Elapsed time: {elapsed:.2f}s")
                generation += 1
    else:
        # Sequential evaluation (original behavior)
        while not es.stop():
            solutions = es.ask()
            fitness_values = [eval_func(x) for x in solutions]
            es.tell(solutions, fitness_values)
            es.disp()
            
            if verbose and generation % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Generation {generation}, Elapsed time: {elapsed:.2f}s")
            generation += 1
    
    if verbose:
        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time:.2f} seconds")
    
    es.result_pretty()

    optimized_params = es.result.xbest
    optimized_MI = es.result.fbest
    print('Optimized_params:', optimized_params)
    print('Optimized_MI:', optimized_MI)
    print('-'*60)

    coefs = np.array(
                     np.dot( optimized_params.reshape( -1, num_basis ),
                            basis_matrices )
                     )

    # These are what Gonzalo was using
    raw_results = {
        "input_coef": coefs[:num_input],
        "output_coef": coefs[num_input:] if optimize_output else None,
        "optimized_params": optimized_params,
        "optimized_MI": optimized_MI,
    }

    # Additional processing from original IT-Pi function
    a_list = [tuple(optimized_params[i*num_basis:(i+1)*num_basis]) for i in range(num_input)]
    coef_pi_list = np.array([np.dot(a, basis_matrices) for a in a_list])
    normalized_coef_pi_list = []
    for coef_pi in coef_pi_list:
        max_abs_value = np.max(np.abs(coef_pi))
        normalized_coef_pi = coef_pi / max_abs_value
        # normalized_coef_pi_list.append(np.round(normalized_coef_pi, 5))
        normalized_coef_pi_list.append(normalized_coef_pi)
        print('coef_pi:', normalized_coef_pi)

    input_list = [calc_pi_omega(np.array(omega), X) for omega in normalized_coef_pi_list]
    input_PI = np.column_stack(input_list)

    output_PI = Y
    if optimize_output:
        normalized_a_list = []
        for a in a_list:
            normalized_a = tuple(x / a[2] for x in a) if len(a) > 2 and a[2] != 0 else tuple(a)
            # normalized_a = np.round(normalized_a,2)
            normalized_a_list.append(normalized_a)
        print('a_list:',normalized_a_list)
        normalized_coef_pi_list = np.array([np.dot(a, basis_matrices) for a in normalized_a_list])
        input_list = np.array([calc_pi(a, basis_matrices, X) for a in normalized_a_list])
        input_PI = np.column_stack(input_list)

        a_list_o = [tuple(optimized_params[i*num_basis+num_para_input:(i+1)*num_basis+num_para_input]) for i in range(num_input)]
        # a_list_o = [np.round(tuple(optimized_params[i*num_basis+num_para_input:(i+1)*num_basis+num_para_input]), 1) for i in range(num_input)]
        # print('a_list_o =', a_list_o)

        modified_a_list_o = []
        first_norm_a = normalized_a_list[0]
        for a_o in a_list_o:
            ratio = a_o[2] / first_norm_a[2] if first_norm_a[2] != 0 else 1.0
            modified_a_o = tuple(a_o[i] - first_norm_a[i] * ratio for i in range(len(a_o)))
            modified_a_list_o.append(modified_a_o)
        a_list_o = modified_a_list_o

        coef_pi_o = np.array(np.dot(a_list_o[0], basis_matrices))
        # coef_pi_o = coef_pi_o / np.max(np.abs(coef_pi_o)) # Don't normalize this cause has to match provided Y!
        # coef_pi_o = np.round(coef_pi_o, 5)
        print('coef_pi_o:', coef_pi_o)
        print("a_list_o:", a_list_o)

        output_PI = np.asarray(np.column_stack([calc_pi(a, basis_matrices, X) for a in modified_a_list_o])).reshape(-1, 1) * Y


    epsilon_values = []
    uq_values = []
    for j in range(input_PI.shape[1]):
        epsilon_full, uq = calculate_bound_and_uq(
            input_PI[:, j].reshape(-1, 1), output_PI, num_trials
        )
        epsilon_values.append(epsilon_full)
        uq_values.append(uq)

    if input_PI.shape[1] > 1:
        epsilon_full, uq = calculate_bound_and_uq(input_PI, output_PI, num_trials)
        epsilon_values.append(epsilon_full)
        uq_values.append(uq)

    return {
        "input_coef": normalized_coef_pi_list,
        "output_coef": coef_pi_o if optimize_output else None,
        "input_coef_basis": normalized_a_list if optimize_output else a_list,
        "output_coef_basis": a_list_o if optimize_output else None,
        "irreducible_error": epsilon_values,
        "uncertainty": uq_values,
        "input_PI": input_PI,
        "output_PI": output_PI,
        "optimized_params": optimized_params,
        "raw_results": raw_results,
    }