import numpy as np
import IT_Pi
from database import db_config as db

__all__ = [
    "tau_data",
]

class tau_data(IT_Pi.ITPi_data):
    # PiY = tau_w/(rho*U_e^2)
    # TODO: Maybe try giving it delta1k too to see if can rebuild Rotta-Clauser parameter identified by 
    # Wenzel as appropriate self-similarity parameter
    _vars_all = ['Ue', 'delta', 'd1', 'd2',
                      'mue', 'dPe', 'dPw', 'rhoe', 'rhow', 'muw']
    _D_in = np.matrix([
        [ 1, 1, 1, 1, -1, -2, -2, -3, -3, -1],  # L
        [ 0, 0, 0, 0,  1,  1,  1,  1,  1,  1],  # m
        [-1, 0, 0, 0, -1, -2, -2,  0,  0, -1],  # t
    ])

    def extract_vars(self, cases:list[db.BL]):
        # TODO: Finish putting this together
        X, Y = np.ndarray([0,self._lvars]), np.ndarray([0,1])
        for c in cases:
            for edge_type in ['delta99', 'delta1', 'delta2', 'delta1k']:
                edge = getattr(c, edge_type)
                if edge is None: 
                    edge, _ = c.find_edge(edge_type)
                    setattr(c, edge_type, edge)
            assert np.all(c.hasdata(('ue','mue','rhoe', 'Bk','Cf','tauw','P', 
                                     'muw', 'rhow')))
            if c.P.ndim > 1 and np.all(c.Bk != 0):
                ide = np.argmin(np.abs(c.y-c.delta99[:,np.newaxis]),axis=-1)
                ide = np.transpose(np.array(list(zip(range(len(c.x)),ide))))

                # Gradient is super sensitive to noise: smooth data first
                # P = gaussian_filter1d(c.P, sigma=2*ide[1,0], axis=0)
                dP = np.gradient(c.P, c.x, axis=0)

                dPe = dP[*ide]
                dPw = dP[:,0]
            else: 
                dPe = dPw = c.Bk*c.tauw/c.delta1k
            Xc = np.vstack((c.ue, c.delta99, c.delta1, c.delta2, c.mue, dPe, dPw, 
                            c.rhoe, c.rhow, c.muw)).T
            Yc = 0.5*c.Cf[:,np.newaxis]
            X, Y = np.vstack((X, Xc)), np.vstack((Y, Yc))
        self.append_data(X,Y)
        return (X, Y)