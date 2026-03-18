from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import matrix_rank
import random
from .IT_PI import calc_basis
from .funcs import norme, return_enew

import pdb

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
    def extract_vars(self, cases:list|tuple, edge_smooth:list[float]|str=None, 
                     grad_smooth:list[float]|str=None,IDs:list=None) -> tuple[np.ndarray, np.ndarray]: ...

    def __init__(self, X=None, Y=None, ID=None) -> None:
        self._lvars = len(self._vars_all)
        if X is not None and Y is not None:
            self._X = X
            self._Y = Y
            self._id = np.array(ID,dtype='T') if ID is not None else \
                np.array(['']*X.shape[0],dtype='T').reshape(-1,1)
        else:
            self._X = np.zeros((0, self._lvars))
            self._Y = np.zeros((0, 1))
            self._id = np.ndarray([0,1],dtype='T')

    def __len__(self):
        return self._id.size

    def append_data(self, X_new:np.ndarray, Y_new:np.ndarray, ID_new=None):
        self._X = np.vstack((self._X, X_new))
        self._Y = np.vstack((self._Y, Y_new))
        if ID_new is not None:
            if isinstance(ID_new, str):
                ID_new = np.array([ID_new]*X_new.shape[0],dtype='T').reshape(-1,1)
            elif isinstance(ID_new, list|np.ndarray):
                ID_new = np.array(ID_new,dtype='T').reshape(-1,1)
            self._id = np.vstack((self._id, ID_new))
        else:
            self._id = np.vstack((self._id, np.array(['']*X_new.shape[0],dtype='T').reshape(-1,1)))

    def get_vars(self, vars:list):
        ii = [self._vars_all.index(x) for x in vars]
        Dmask = ~np.all(np.asarray(self._D_in[:, ii]) == 0, axis=1)
        return self._X[:, ii], self._Y, self._id, self._D_in[:, ii][Dmask]
    
    def get_data(self, vars:list, Npts:int = 2000):
        X, Y, IDs, D_in = self.get_vars(vars)
        try:
            rind = random.sample(range(X.shape[0]), Npts)
        except ValueError: # Asked for more points than we have, just return all data
            rind = np.ones(X.shape[0], dtype=bool)
        num_basis = D_in.shape[1] - matrix_rank(D_in)
        basis_matrices = calc_basis(D_in, num_basis)
        return X[rind], Y[rind], IDs[rind], basis_matrices
    
    def split_train_valid(self, train_ratio:float = 0.8):
        rind = np.random.random(self._X.shape[0]) < train_ratio
        myclass = type(self)
        train_data = myclass(X=self._X[rind], Y=self._Y[rind], ID=self._id[rind])
        valid_data = myclass(X=self._X[~rind], Y=self._Y[~rind], ID=self._id[~rind])
        return train_data, valid_data

def get_exp(ITPi_r:dict|str, D_in:np.matrix|np.ndarray, Vars:list[str]|tuple[str]=None, exp_thresh:float=0.01,
            inorm:int=None):
    if isinstance(ITPi_r,str):
        ITPi_r = np.load(ITPi_r, allow_pickle=True)
    # Normalize exponents and remove small terms (adjusting rest to preserve
    # non-dimensionality)
    eorig_in = norme(ITPi_r['input_coef'], 5, inorm=inorm)
    output_coef = ITPi_r['output_coef']
    # Don't normalize output exponents as must multiply by Y!
    eorig_out = np.round(output_coef, 5) if None not in output_coef else None
    e_in = np.zeros_like(eorig_in)
    e_out = np.zeros_like(eorig_out) if eorig_out is not None else None
    if Vars is not None and 'dPe' in Vars:
        # flip eorig so dPe ~ 0 is always the first vector 
        # - SH: I think this is just so first Pi group doesn't have dPe?
        eorig_in = np.roll(eorig_in, eorig_in[:, Vars.index('dPe')].argmin(), axis=0)
        eorig_out = np.roll(eorig_out, eorig_out[:, Vars.index('dPe')].argmin(), axis=0) if e_out is not None else None
    for k, e_ in enumerate(eorig_in):
        inds = np.where(np.abs(eorig_in[k]) < exp_thresh)[0]
        e_in[k] = return_enew(np.array(D_in), e_, inds)
    if e_out is not None:
        for k, e_ in enumerate(eorig_out):
            inds = np.where(np.abs(eorig_out[k]) < exp_thresh)[0]
            e_out[k] = return_enew(np.array(D_in), e_, inds)
    return np.array(e_in), np.array(e_out) if e_out is not None else None

def pretty_exps(e:np.ndarray, varlbls:list[str]|tuple[str], prnt:bool=False):
    labels = []
    for i, e_ in enumerate(e):
        num, den = "", ""
        for var, exp in zip(varlbls, e_):
            if np.abs(exp) < 1e-3: continue
            if '$' in var: var = var.replace('$','')
            if exp < 0:
                den += f"{var}^{{{-exp:.3f}}}"
            elif exp > 0:
                num += f"{var}^{{{exp:.3f}}}"
        if num == "": num = "1"
        if den == "": labels.append(f"${num}$")
        else: labels.append(f"$\\frac{{{num}}}{{{den}}}$")
        if prnt: print(f"\\Pi_{i+1} = {labels[-1][1:-1]}")
    return labels

def rescale_output(results:dict, basis_matrices:np.ndarray|np.matrix, iy:int) -> None:
    # Remove contribution of input Pi group to cancel out output group dependence
    # iy is the index of the variable with non-physical dependence (e.g. y)
    output_coef_old = results['output_coef'].copy()
    ibs = np.argwhere(basis_matrices[:,iy]!=0)[...,0] # Basis vectors with dependence
    for a_o in results['a_list_o']:
        norm = np.sum(results['a_list_i'][0][ibs] * np.asarray(basis_matrices)[ibs,iy])
        ratio = np.sum(a_o[ibs] * np.asarray(basis_matrices)[ibs,iy]) / norm if np.abs(norm)>0.1 else 1.0
        a_o -= results['a_list_i'][0] * ratio
    results['output_coef'] = np.array(np.dot(results['a_list_o'], basis_matrices))
    
    # Check rescaling is consistent (not changing solution structure)
    scaling = (output_coef_old - results['output_coef']).squeeze()/results['input_coef'][0]
    assert np.allclose(scaling, scaling[0]), "Output rescaling is not consistent across basis vectors, check results"