import numpy as np
import random
from cma import CMAEvolutionStrategy
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from .IT_PI import MI_input_output

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

    return {
        "input_coef": coefs[:num_input],
        "output_coef": coefs[num_input:] if optimize_output else None,
        "optimized_params": optimized_params,
        "optimized_MI": optimized_MI,
    }

