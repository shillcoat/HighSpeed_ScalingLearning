import numpy as np
import IT_Pi
from database import db_config as db
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

from database.db_config import BL

import pdb

__all__ = [
    "tau_data",
    "U_data",
    "interpolate_profiles"
]

def interpolate_profiles(
    x: np.ndarray,
    y: np.ndarray,
    props: list[np.ndarray],
    n_y: int = 200,
    y_min_factor: float = 1.0,
    log_base: float = 10.0, ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Written by Claude
    Interpolate turbulent boundary layer velocity profiles u(x, y) and
    resample the wall-normal coordinate on a logarithmic grid.

    Parameters
    ----------
    x : (Nx,) array
        Streamwise coordinate values.
    y : (Ny,) array
        Wall-normal coordinate values (must be strictly positive).
    props : list of (Nx, Ny) arrays
        Property fields to interpolate, where props[k][i, j] = props[k](x[i], y[j]).
    n_y : int
        Number of points in the new logarithmic y-grid.
    y_min_factor : float
        Scaling factor on y[0] to set the start of the log grid.
    log_base : float
        Base of the logarithm used to space the new y-grid (default 10).

    Returns
    -------
    x : (Nx,) array
        Original streamwise coordinates (unchanged).
    y_log : (n_y,) array
        New logarithmically-spaced wall-normal coordinates.
    props_log : list of (Nx, n_y) arrays
        Property fields resampled onto (x, y_log).
    """
    # Build the logarithmic y-grid
    y_start = (y[1] if y[0] == 0 else y[0]) * y_min_factor
    y_end = y[-1]

    log_start = np.log(y_start) / np.log(log_base)
    log_end = np.log(y_end) / np.log(log_base)
    y_log = np.logspace(log_start, log_end, num=n_y, base=log_base)

    # Interpolate
    props_log = []
    for prop in props:
        prop_log = np.zeros((x.size, n_y))
        for i in range(x.size):
            prop_log[i, :] = interp1d(y, prop[i, :], kind="cubic", bounds_error=False, 
                                      fill_value="extrapolate")(y_log)
        props_log.append(prop_log)

    return x, y_log, props_log

class tau_data(IT_Pi.ITPi_data):
    # TODO: Maybe try giving it delta1k too to see if can rebuild Rotta-Clauser parameter identified by 
    # Wenzel as appropriate self-similarity parameter
    # First 7 variables must remain in this order for compatibility with Gonzalo's data
    _vars_all = ['Ue', 'delta', 'd1', 'd2',
                      'mue', 'dPe', 'dPw', 'rhoe', 'rhow', 'muw']
    _D_in = np.matrix([
        [ 1, 1, 1, 1, -1, -2, -2, -3, -3, -1],  # L
        [ 0, 0, 0, 0,  1,  1,  1,  1,  1,  1],  # m
        [-1, 0, 0, 0, -1, -2, -2,  0,  0, -1],  # t
    ])

    def extract_vars(self, cases:list[db.BL], edge_smooth:list[float]|str=None, 
                     grad_smooth:list[float]|str=None, IDs:list=None):
        # Watch out for edge artifacts when smoothing gradients!
        X, Y = np.ndarray([0,self._lvars]), np.ndarray([0,1])
        ID_new = np.ndarray([0,1],dtype='T')
        for ic, c in enumerate(cases):
            for edge_type in ['delta99', 'delta1', 'delta2', 'delta1k']:
                edge = getattr(c, edge_type)
                if edge is None: 
                    edge, _ = c.find_edge(edge_type, sigma_smooth=edge_smooth)
                    setattr(c, edge_type, edge)
            assert np.all(c.hasdata(('ue','mue','rhoe', 'Bk','Cf','tauw','P', 
                                     'muw', 'rhow')))
            if c.P.ndim > 1 and np.all(c.Bk != 0):
                ide = np.argmin(np.abs(c.y-c.delta99[:,np.newaxis]),axis=-1)
                ide = np.transpose(np.array(list(zip(range(len(c.x)),ide))))

                P = c.P
                if grad_smooth is not None:
                    P = gaussian_filter(P, sigma=grad_smooth)
                dP = np.gradient(P, c.x, axis=0)

                dPe = dP[*ide]
                dPw = dP[:,0]
            else: 
                dPe = dPw = c.Bk*c.tauw/c.delta1k
            Xc = np.vstack((c.ue, c.delta99, c.delta1, c.delta2, c.mue, dPe, dPw, 
                            c.rhoe, c.rhow, c.muw)).T
            Yc = 0.5*c.Cf[:,np.newaxis]
            if IDs is None:
                IDc = np.array(['']*Xc.shape[0],dtype='T').reshape(-1,1)
            else:
                IDc = np.array([IDs[ic]]*Xc.shape[0],dtype='T').reshape(-1,1)
            X, Y = np.vstack((X, Xc)), np.vstack((Y, Yc))
            ID_new = np.vstack((ID_new, IDc))
        self.append_data(X,Y,ID_new=ID_new)
        return (X, Y)
    

class U_data(IT_Pi.ITPi_data):
    _vars_all = ['Ue', 'delta', 'utau', 'y', 'mu', 'muw', 'rho', 'rhow', 'dPe']
    _D_in = np.matrix([
        # Ue d utau y  mu muw rho rhow dP
        [ 1, 1,  1, 1, -1, -1, -3, -3, -2],  # L
        [ 0, 0,  0, 0,  1,  1,  1,  1,  1],  # m
        [-1, 0, -1, 0, -1, -1,  0,  0, -2],  # t
    ])

    def extract_vars(self, cases:list[db.BL], edge_smooth:list[float]|str=None, grad_smooth:list[float]|str=None, IDs:list=None,
                     remove_wake:bool=True, resample:bool=True, limitx:bool=True):
        # Includes option to resample on log scale to give training more near-wall data points
        X, Y = np.ndarray([0,self._lvars]), np.ndarray([0,1])
        ID_new = np.ndarray([0,1],dtype='T')
        for ic, c in enumerate(cases):
            # If have a lot of profiles in 'x' only take the middle 10 to avoid overwhelming other datasets
            xslice = np.s_[c.x.size//2:c.x.size//2+10] if (c.x.size > 20 and limitx) else np.s_[:]
            for edge_type in ['delta99','delta1k']:
                try:
                    edge = getattr(c, edge_type)
                except AttributeError:
                    edge = getattr(c, 'h') # Messy fix to support channel data for now
                if edge is None: 
                    edge, _ = c.find_edge(edge_type, sigma_smooth=edge_smooth)
                    setattr(c, edge_type, edge)
            assert np.all(c.hasdata(('ue','mu','rho','utau','P','muw','rhow','y','u')))
            
            if isinstance(c,BL):
                ide = np.argmin(np.abs(c.y-c.delta99[:,np.newaxis]),axis=-1)
                ide2D = np.transpose(np.array(list(zip(range(len(c.x)),ide+1))))

            if isinstance(c,BL):
                assert c.hasdata(('Bk','tauw'))
                if c.P.squeeze().ndim > 1 and np.all(c.Bk != 0):
                    P = c.P
                    if grad_smooth is not None:
                        P = gaussian_filter(P, sigma=grad_smooth)
                    dP = np.gradient(P, c.x, axis=0)

                    dPe = dP[*ide2D]
                else: 
                    dPe = c.Bk*c.tauw/c.delta1k
            else: dPe = np.zeros_like(c.x)

            if resample:
                props = [c.u, c.rho, c.mu]
                _, ynew, props_log = interpolate_profiles(c.x, c.y, props, n_y=len(c.y))
                c.y = ynew
                c.u, c.rho, c.mu = props_log
                if isinstance(c, BL):
                    ide = np.argmin(np.abs(c.y-c.delta99[:,np.newaxis]),axis=-1)

            c.x = c.x[xslice]
            if isinstance(c, BL):
                ide = ide[xslice]

            if remove_wake and isinstance(c, BL):
                y = np.hstack([c.y[:i+1] for i in ide])
                rhow = np.hstack([[c.rhow[xslice][i]]*(ide[i]+1) for i in range(len(ide))])
                muw = np.hstack([[c.muw[xslice][i]]*(ide[i]+1) for i in range(len(ide))])
                Ue = np.hstack([[c.ue[xslice][i]]*(ide[i]+1) for i in range(len(ide))])
                delta = np.hstack([[c.delta99[xslice][i]]*(ide[i]+1) for i in range(len(ide))])
                dPe = np.hstack([[dPe[xslice][i]]*(ide[i]+1) for i in range(len(ide))])
                utau = np.hstack([[c.utau[xslice][i]]*(ide[i]+1) for i in range(len(ide))])

                u = np.hstack([c.u[xslice][i,:ide[i]+1] for i in range(len(ide))])
                mu = np.hstack([c.mu[xslice][i,:ide[i]+1] for i in range(len(ide))])
                rho = np.hstack([c.rho[xslice][i,:ide[i]+1] for i in range(len(ide))])
            else:
                y = np.tile(c.y, (len(c.x),1)).ravel()
                rhow = np.tile(c.rhow[xslice], (1,len(c.y))).ravel()
                muw = np.tile(c.muw[xslice], (1,len(c.y))).ravel()
                Ue = np.tile(c.ue[xslice], (1,len(c.y))).ravel()
                delta = np.tile(c.delta99[xslice] if isinstance(c,BL) else c.h, (1,len(c.y))).ravel()
                dPe = np.tile(dPe[xslice], (1,len(c.y))).ravel()
                utau = np.tile(c.utau[xslice], (1,len(c.y))).ravel()

                # Flatten rather than raveling to ensure produces copy
                u = c.u[xslice].flatten()
                mu = c.mu[xslice].flatten()
                rho = c.rho[xslice].flatten()

            Xc = np.vstack((Ue, delta, utau, y, mu, muw, rho, rhow, dPe)).T
            Yc = (u/utau)[:,np.newaxis]

            if IDs is None:
                IDc = np.array(['']*Xc.shape[0],dtype='T').reshape(-1,1)
            else:
                IDc = np.array([IDs[ic]]*Xc.shape[0],dtype='T').reshape(-1,1)
            X, Y = np.vstack((X, Xc)), np.vstack((Y, Yc))
            ID_new = np.vstack((ID_new, IDc))
        self.append_data(X,Y,ID_new=ID_new)
        return (X, Y)
    

class U_data_recast(U_data):
    # Recast the problem to better match existing scaling laws: y and u are no longer provided and
    # the output becomes 1/utau
    _vars_all = ['dudy', 'y', 'rho', 'rhow', 'mu', 'muw', 'utau']
    _D_in = np.matrix([  # Lazy dimensionality reduction: know that mu and muw will have to balance each other and likewise for rho
        # dudy y  rho rhow mu  muw utau
        [  0,  1, -3,  -3, -1, -1,  1], # L
        [  0,  0,  1,   1,  1,  1,  0], # m
        [ -1,  0,  0,   0, -1, -1, -1], # t
    ])

    def __init__(self, ypref=None, upref=None, ref_prof=None, X=None, Y=None, ID=None) -> None:
        super().__init__(X, Y, ID)
        if ypref is None and upref is None and ref_prof is None:
            raise Exception("Must provide either ypref and upref or ref_prof to define the reference velocity \
                             profile for calculating Y")
        if ref_prof is None:
            self.ref_prof = interp1d(ypref.squeeze(), upref.squeeze(), kind="cubic", 
                                    bounds_error=False, fill_value="extrapolate")
        else:
            self.ref_prof = ref_prof

    def split_train_valid(self, train_ratio:float = 0.8):
        rind = np.random.random(self._X.shape[0]) < train_ratio
        myclass = type(self)
        train_data = myclass(ref_prof=self.ref_prof, X=self._X[rind], Y=self._Y[rind], ID=self._id[rind])
        valid_data = myclass(ref_prof=self.ref_prof, X=self._X[~rind], Y=self._Y[~rind], ID=self._id[~rind])
        return train_data, valid_data
    
    def get_Y(self, y, rhow, muw, utau):
        yplus = y*rhow*utau/muw
        uplus = self.ref_prof(yplus)
        return np.gradient(uplus,yplus)

    def extract_vars(self, cases:list[db.BL], edge_smooth:list[float]|str=None, grad_smooth:list[float]|str=None, IDs:list=None,
                     remove_wake:bool=True, resample:bool=True):
        # Includes option to resample on log scale to give training more near-wall data points
        X, Y = np.ndarray([0,self._lvars]), np.ndarray([0,1])
        ID_new = np.ndarray([0,1],dtype='T')
        for ic, c in enumerate(cases):
            for edge_type in ['delta99','delta1k']:
                edge = getattr(c, edge_type)
                if edge is None: 
                    edge, _ = c.find_edge(edge_type, sigma_smooth=edge_smooth)
                    setattr(c, edge_type, edge)
            assert np.all(c.hasdata(('mu','rho','utau','muw','rhow')))
            
            ide = np.argmin(np.abs(c.y-c.delta99[:,np.newaxis]),axis=-1)

            if resample:
                props = [c.rho, c.mu]
                _, ynew, props_log = interpolate_profiles(c.x, c.y, props, n_y=len(c.y))
                c.y = ynew
                c.rho, c.mu = props_log
                ide = np.argmin(np.abs(c.y-c.delta99[:,np.newaxis]),axis=-1)

            if remove_wake:
                y = np.hstack([c.y[:i+1] for i in ide])
                rhow = np.hstack([[c.rhow[i]]*(ide[i]+1) for i in range(len(ide))])
                muw = np.hstack([[c.muw[i]]*(ide[i]+1) for i in range(len(ide))])
                utau = np.hstack([[c.utau[i]]*(ide[i]+1) for i in range(len(ide))])

                dupdyp_ref = []
                for i, ide_ in enumerate(ide):
                    dupdyp_ref.append(self.get_Y(c.y[:ide_+1], c.rhow[i], c.muw[i], c.utau[i]))

                mu = np.hstack([c.mu[i,:ide[i]+1] for i in range(len(ide))])
                rho = np.hstack([c.rho[i,:ide[i]+1] for i in range(len(ide))])

                dudy = np.hstack([np.gradient(c.u[i,:ide[i]+1], c.y[:ide[i]+1]) \
                                  for i in range(len(ide))])
            else:
                y = np.tile(c.y, (len(c.x),1)).ravel()
                rhow = np.tile(c.rhow, (1,len(c.y))).ravel()
                muw = np.tile(c.muw, (1,len(c.y))).ravel()
                utau = np.tile(c.utau, (1,len(c.y))).ravel()

                dupdyp_ref = []
                for i in range(len(c.x)):
                    dupdyp_ref.append(self.get_Y(c.y, c.rhow[i], c.muw[i], c.utau[i]))

                # Flatten rather than raveling to ensure produces copy
                mu = c.mu.flatten()
                rho = c.rho.flatten()

                dudy = np.gradient(c.u, c.y, axis=-1).flatten()

            Xc = np.vstack((dudy, y, rho, rhow, mu, muw, utau)).T
            Yc = np.array(dupdyp_ref).reshape(-1,1)

            if IDs is None:
                IDc = np.array(['']*Xc.shape[0],dtype='T').reshape(-1,1)
            else:
                IDc = np.array([IDs[ic]]*Xc.shape[0],dtype='T').reshape(-1,1)
            X, Y = np.vstack((X, Xc)), np.vstack((Y, Yc))
            ID_new = np.vstack((ID_new, IDc))
        self.append_data(X,Y,ID_new=ID_new)
        return (X, Y)

    
class Yuan_U_data(U_data):
    _vars_all = ["y", "rho", "mu", "rhow", "muw", "utau"]
    _D_in = np.matrix('1 -3 -1 -3 -1  1; ' \
                      '0  0 -1  0 -1 -1; ' \
                      '0  1  1  1  1  0')
    
    def get_data(self, vars:list, Npts:int = 2000):
        X, Y, IDs, basis_matrices = super().get_data(vars, Npts)
        basis_matrices[2,:] = -basis_matrices[2,:]  # Don't know why we do this, doesn't seem to change much
        return X, Y, IDs, basis_matrices