"""
Script to read in data files from Volpiani et al. (2020) and convert to database format
Adapted from that provided by Volpiani: 
https://github.com/psvolpiani/2020-PRF-transformation/blob/master/Trasformation/plot_extract_bl_profiles_all.py

---------------------------------------------------------------------------------------------------

"Volpiani, P. S., Bernardini, M., & Larsson, J. (2018). 
Effects of a nonadiabatic wall on supersonic shock/boundary-layer interactions. 
Physical Review Fluids, 3(8), 083401."
"""

# %% Imports
import numpy as np
from glob import glob

from db_config import BL, save_case
from filepaths import Volpiani2020_path  # This file is not kept in repo

# %% Fetch and iterate through cases
cases = glob(Volpiani2020_path + "/*/*.stats")
# Case parameter info not in file
# [muref, muexp, Tw, Tr, xplot, xsp, dz]
c_params = {
    "m5_twtr190_bl":        [1.2e-5,    0.75,   10.355, 10.355/1.9, 40.0,   65, 5.0/140.0],
    "m5_twtr190_bl_m-025":  [5.0e-5,    -0.25,  10.355, 10.355/1.9, 45.0,   55, 5.0/200.0],
    "m5_twtr080_bl_hRe":    [1.8e-5,    0.75,   4.36,   4.36/0.8,   40.0,   65, 5.0/200.0],
    "m5_twtr080_bl_mhRe":   [1.2e-5,    0.75,   4.36,   4.36/0.8,   40.0,   65, 5.0/300.0],
    "m5_twtr080_bl_vhRe":   [0.9e-5,    0.75,   4.36,   4.36/0.8,   40.0,   65, 5.0/340.0],
    "m228_twtr050_bl":      [1.29e-4,   0.75,   0.963,  0.963/0.5,  35.0,   60, 5.0/256.0],
    "m228_twtr100_bl":      [1.22e-4,   0.75,   1.920,  1.920,      35.0,   60, 5.0/128.0],
    "m228_twtr100_bl_hRe":  [0.6e-4,    0.75,   1.920,  1.920,      35.0,   60, 5.0/256.0],
    "m228_twtr190_bl":      [1.20e-4,   0.75,   3.658,  3.658/1.9,  35.0,   60, 5.0/180.0]
}

for c in cases:
    dbCase = BL('nondim', incomp=0, chem=0, gamma=1.4, Pr=0.7, Bk=0)
    casename = c.split('/')[-2]
    muref, muexp, dbCase.Tw, dbCase.Tr, xplot, xsp, dz = c_params[casename]

    # Open and read binary Hybrid file
    with open(c, 'rb') as f:
        while b'%%%' not in f.readline():
            pass
        arg = np.fromfile(f, count=3, dtype='int32')
        nx, ny, nv = arg[0], arg[1], arg[2]
        dbCase.x = np.fromfile(f, count=nx, dtype='float64')
        dbCase.y = np.fromfile(f, count=ny, dtype='float64')
        avg = np.fromfile(f, count=nx*ny*nv, dtype='float64')
        avg = avg.reshape([nx,ny,nv], order='F')

    dbCase.mu_law = lambda T: muref * T**muexp
    dbCase.uinf = 1.0
    dbCase.rhoinf = 1.0
    dbCase.Tinf = 1.0
    
    # Reynolds mean and fluctuating quantities
    dbCase.rho = avg[:,:,0]
    dbCase.u = avg[:,:,1]
    dbCase.v = avg[:,:,2]
    dbCase.w = avg[:,:,3]
    dbCase.P = avg[:,:,4]
    dbCase.T = avg[:,:,5]
    dbCase.mu = dbCase.mu_law(dbCase.T)
    dbCase.rhoprhop = avg[:,:,15]
    dbCase.PpPp = avg[:,:,16]

    # Favre mean quantities
    dbCase.u_F = avg[:,:,10]
    dbCase.v_F = avg[:,:,11]
    dbCase.w_F = avg[:,:,12]
    dbCase.T_F = avg[:,:,13]
    dbCase.mu_F = dbCase.mu_law(dbCase.T_F)
    dbCase.R = np.mean(dbCase.P/dbCase.rho/dbCase.T_F)
    dbCase.a = np.sqrt(dbCase.gamma*dbCase.R*dbCase.T)
    dbCase.M = dbCase.u / dbCase.a
    dbCase.Minf = dbCase.uinf / np.mean(dbCase.a[:,-1])

    # Favre fluctuating quantities
    dbCase.ruppupp = dbCase.rho * avg[:,:,21]
    dbCase.rvppvpp = dbCase.rho * avg[:,:,22]
    dbCase.rwppwpp = dbCase.rho * avg[:,:,23]
    dbCase.rTppTpp = dbCase.rho * avg[:,:,24]
    dbCase.ruppvpp = dbCase.rho * avg[:,:,25]
    dbCase.ruppwpp = dbCase.rho * avg[:,:,26]
    dbCase.rvppwpp = dbCase.rho * avg[:,:,27]
    dbCase.ruppTpp = dbCase.rho * avg[:,:,28]
    dbCase.rvppTpp = dbCase.rho * avg[:,:,29]
    dbCase.rwppTpp = dbCase.rho * avg[:,:,30]

    del avg  # Done reading in so clear up memory

    # Wall quantities
    dbCase.muw = dbCase.mu_law(dbCase.Tw)
    dbCase.rhow = dbCase.P[:,0]/dbCase.Tw/dbCase.R
    # Use mean mu between wall and first cell for calculating wall props
    mu_eff = 0.5 * (dbCase.muw + dbCase.mu_law(dbCase.T[:,0]))
    cp = dbCase.gamma*dbCase.R/(dbCase.gamma-1)
    dbCase.tauw = mu_eff * dbCase.u[:,0] / dbCase.y[0]
    dbCase.qw = cp * mu_eff / dbCase.Pr * (dbCase.T[:,0]-dbCase.Tw)/dbCase.y[0]
    
    dbCase.utau = np.sqrt(np.abs(dbCase.tauw) / dbCase.rhow)
    dbCase.uplus = dbCase.u/np.transpose([dbCase.utau])
    dbCase.Mtau = dbCase.utau / np.sqrt(dbCase.gamma*dbCase.R*dbCase.Tw)
    dbCase.deltaplus = dbCase.muw / (dbCase.rhow * dbCase.utau)
    dbCase.yplus = dbCase.y/np.transpose([dbCase.deltaplus])
    mu_muw = dbCase.mu/dbCase.muw
    rho_rhow = dbCase.rho/np.transpose([dbCase.rhow])
    dbCase.deltastar = mu_muw*np.sqrt(1/rho_rhow)*np.transpose([dbCase.deltaplus])
    dbCase.ystar = dbCase.y/dbCase.deltastar

    dbCase.Cf = 2*dbCase.tauw/(dbCase.rhoinf + dbCase.uinf**2)
    dbCase.Bq = dbCase.qw/(dbCase.rhow*cp*dbCase.utau*dbCase.Tw)

    # Get visual thickness
    dbCase.delta99, id99 = dbCase.find_edge()
    # Friction Reynolds numbers
    dbCase.Retau = dbCase.delta99 / dbCase.deltaplus
    id99 = np.transpose(np.array(list(zip(range(nx),id99))))
    dbCase.Retaustar = dbCase.delta99 / dbCase.deltastar[*id99]
    
    cname = c.split('/')[-2]
    ix = np.where(dbCase.x>=xplot)[0][0]
    print(f"Case: {cname}\nRe_tau = {dbCase.Retau[ix]:.4f}\nRe_tau* = {dbCase.Retaustar[ix]:.4f}\n")
    
    i = c[::-1].index('/') + 1
    save_case(dbCase, c[:-i] + ".dill")
# %%
