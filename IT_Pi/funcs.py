import numpy as np
from scipy.optimize import least_squares
from .IT_PI import calculate_bound_and_uq
from itertools import combinations

__all__ = [
    "getPiIfromXe",
    "norme",
    "return_enew",
    "computeIrrError",
]


def getPiIfromXe(X_, e_, tol: float = 1e-10):
    """Compute a product of variables raised to given exponents, with sign
    handling.

    This function evaluates a generalized monomial of the form

        Π_i sign(X_i) * |X_i|^{e_i}

    across the last dimension of the input, while avoiding sign issues
    when exponents are (numerically) zero.

    Parameters
    X_ : ndarray
        Input array of shape (..., M), where M is the number of variables.
        Each row (or trailing dimension) represents a set of values X_i.

    e_ : ndarray
        Array of exponents with shape broadcastable to X_. Exponents define
        the powers applied to each variable.

    Returns
        ndarray: Array of shape (...) containing the product along the last axis:
        (sign(X) * |X|**e).prod(-1).

    Notes
    - For entries where |e_i| < 1e-10, the sign is forced to +1 to prevent
      undefined behavior when raising negative numbers to non-integer
      exponents.
    - This effectively implements the Π-group construction in dimensional
      analysis or monomial basis functions in regression.
    """
    mysign = np.sign(X_)
    mysign[:, np.abs(e_) < tol] = 1
    return (mysign * np.abs(X_) ** e_).prod(-1)


def norme(e_, prec=3):
    return np.round(e_ / np.abs(e_).max(1)[:, None], prec)


def return_enew(D, e_, fixed_indices):

    n_vars = len(e_)
    free_indices = [i for i in range(n_vars) if i not in fixed_indices]

    def objective(x_free):
        """Minimize change from original while maintaining dimensionless constraint"""
        x = np.zeros(n_vars)
        x[free_indices] = x_free
        x[fixed_indices] = 0.0

        # Dimensional constraint (must be zero)
        dim_violation = D @ x

        # Also minimize deviation from original (weighted less)
        deviation = x_free - e_[free_indices]

        return np.concatenate([dim_violation, 0.01 * deviation])

    # Initial guess
    x0 = e_[free_indices]

    # Solve
    result = least_squares(objective, x0)

    # Build final solution
    new_exponents = np.zeros_like(e_)
    new_exponents[free_indices] = result.x
    return new_exponents
    # new_exponents[fixed_indices] = 0.0

    # print("\nOriginal exponents:")
    # print(original_exponents)
    # print("\nNew exponents (with last two = 0):")
    # print(new_exponents)
    # print("\nDimensional check (should be ~[0, 0]):")


def computeIrrError(input_PI, output_PI, num_trials):
    """
    Compute irreducibility error bounds for all possible subsets of input dimensions.

    This function evaluates the information-theoretic error bound for every
    combination of input variables, from single variables up to the full set.
    It systematically tests which subsets of inputs carry sufficient information
    to predict the output, enabling identification of minimal sufficient
    representations.

    Parameters
    ----------
    input_PI : ndarray of shape (ndim, Nsamples)
        Input data matrix where each row represents a different input dimension

    output_PI : ndarray of shape (Nsamples, 1)
        Output data vector corresponding to the target quantity.

    num_trials : int
        Number of bootstrap trials for uncertainty quantification.

    Returns
    -------
    epsilon_values : list of float
        Irreducibility error bounds for each input subset. Lower values indicate
        the subset captures more information about the output.
    uq_values : list of float
        Uncertainty quantification (e.g., standard error) for each error bound,
        estimated via bootstrap resampling.
    ids : list of tuple
        Index tuples identifying which input dimensions were used for each
        corresponding entry in epsilon_values and uq_values. Ordered by subset
        size (all singletons first, then pairs, etc.).
    """
    epsilon_values = []
    uq_values = []
    ids = []

    nin = input_PI.shape[0]

    # Loop over all subset sizes: 1, 2, 3
    for size in range(1, nin + 1):
        for indices in combinations(range(nin), size):
            epsilon_full, uq = calculate_bound_and_uq(
                input_PI[list(indices)].T, output_PI, num_trials
            )
            epsilon_values.append(epsilon_full)
            uq_values.append(uq)
            ids.append(indices)

    return epsilon_values, uq_values, ids
