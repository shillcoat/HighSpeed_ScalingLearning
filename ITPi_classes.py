import numpy as np
import IT_Pi
from database import db_config as db
from scipy.ndimage import gaussian_filter

__all__ = [
    "tau_data",
    "U_data",
]

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

                # Gradient is super sensitive to noise: smooth data first
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
    # TODO: Currently more or less just a copy of tau_data with different variable names. Need to modify.
    _vars_all = ['Ue', 'u', 'delta', 'd1', 'd2', 'y',
                      'mue', 'dPe', 'dPw', 'rhoe', 'rhow', 'muw']
    _D_in = np.matrix([
        [ 1,  1, 1, 1, 1, 1, -1, -2, -2, -3, -3, -1],  # L
        [ 0,  0, 0, 0, 0, 0,  1,  1,  1,  1,  1,  1],  # m
        [-1, -1, 0, 0, 0, 0, -1, -2, -2,  0,  0, -1],  # t
    ])

    def extract_vars(self, cases:list[db.BL], edge_smooth:list[float]|str=None, grad_smooth:list[float]|str=None, IDs:list=None):
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

                # Gradient is super sensitive to noise: smooth data first
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