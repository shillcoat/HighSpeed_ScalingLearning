from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import matrix_rank
import random
from .IT_PI import calc_basis
from .funcs import norme, return_enew

# Create as abstract class so that error is raised if properties/methods are not defined
# by subclass
class ITPi_data(ABC):
    @property
    @abstractmethod
    def _vars_all(self) -> list: ...

    @property
    @abstractmethod
    def _D_in(self) -> np.matrix: ...

    @abstractmethod
    def extract_vars(self, cases:list|tuple) -> tuple[np.ndarray, np.ndarray]: ...

    def __init__(self, X=None, Y=None) -> None:
        self._lvars = len(self._vars_all)
        if X is not None and Y is not None:
            self._X = X
            self._Y = Y
        else:
            self._X = np.zeros((0, self._lvars))
            self._Y = np.zeros((0, 1))

    def append_data(self, X_new:np.ndarray, Y_new:np.ndarray):
        self._X = np.vstack((self._X, X_new))
        self._Y = np.vstack((self._Y, Y_new))

    def get_data(self, vars:list):
        ii = [self._vars_all.index(x) for x in vars]
        return self._X[:, ii], self._Y, self._D_in[:, ii]
    
    def get_traindata(self, vars:list, Ntrain:int = 2000):
        X, Y, D_in = self.get_data(vars)
        rind = random.sample(range(X.shape[0]), Ntrain)
        num_basis = D_in.shape[1] - matrix_rank(D_in)
        basis_matrices = calc_basis(D_in, num_basis)
        return X[rind], Y[rind], basis_matrices

    def get_validdata(self, vars:list, Nvalid:int = 2000):
        X, Y, _ = self.get_data(vars)
        rind = random.sample(range(X.shape[0]), Nvalid)
        return X[rind], Y[rind]

def get_exp(ITPi_r:dict|str, D_in:np.matrix|np.ndarray, Vars:list[str]|tuple[str]=None, exp_thresh:float=0.01):
    if isinstance(ITPi_r,str):
        ITPi_r = np.load(ITPi_r, allow_pickle=True)
    # Normalize exponents and remove small terms (adjusting rest to preserve
    # non-dimensionality)
    eorig = norme(ITPi_r['input_coef'], 5)
    e = np.zeros_like(eorig)
    if Vars is not None:
        # flip eorig so dPe ~ 0 is always the first vector 
        # - SH: I think this is just so first Pi group doesn't have dPe?
        eorig = np.roll(eorig, eorig[:, Vars.index('dPe')].argmin(), axis=0)
    for k, e_ in enumerate(eorig):
        inds = np.where(np.abs(eorig[k]) < exp_thresh)[0]
        e[k] = return_enew(np.array(D_in), e_, inds)
    return np.array(e)

def pretty_exps(e:np.ndarray, varlbls:list[str]|tuple[str], prnt:bool=False):
    labels = []
    for e_ in e:
        num, den = "", ""
        for var, exp in zip(varlbls, e_):
            if exp < 0:
                den += f"{var}^{{{-exp:.3f}}}"
            elif exp > 0:
                num += f"{var}^{{{exp:.3f}}}"
        if num == "": num = "1"
        if den == "": labels.append(f"${num}$")
        else: labels.append(f"$\\frac{{{num}}}{{{den}}}$")
        if prnt: print(labels[-1])
    return labels