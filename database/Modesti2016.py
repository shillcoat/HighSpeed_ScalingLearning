"""
Script to read in data files from Modesti and Pirozzoli (2016) and convert to database format

A little quick and dirty for now as I'm limited on time

---------------------------------------------------------------------------------------------------

 D. Modesti and S. Pirozzoli,  "Reynolds and Mach number effects in compressible turbulent channel flow", 
   International Journal of Heat and Fluid Flow (2016)
"""

# %% Imports
import numpy as np
from glob import glob

from db_config import Channel, save_case
from filepaths import Modesti2016_path  # This file is not kept in repo
from prop_checks import check_edges

# %% Fetch and iterate through cases
cases = glob(Modesti2016_path + "/CH*.dat")
header_size = 17
for c in cases:
    cname = c.split('/')[-1][:-4]
    dbCase = Channel('nondim', incomp=0, chem=0, gamma=1.4, Pr=0.71, x=0.0)

    with open(c,'r') as f:
        for i, line in enumerate(f):
            if i == 1:
                dbCase.Mbulk = float(line.split()[-1])
            if i == 3:
                dbCase.Cf = float(line.split()[-1])
            if i == 7:
                dbCase.Retau = float(line.split()[-1])
            if i == 8:
                dbCase.Rebulk = float(line.split()[-1])
            if i == 12:
                dbCase.Bq = -float(line.split()[-1])
            if i == header_size: break

    # Come back in later and add thermodynamics

    dat = np.loadtxt(c, skiprows=header_size)
    dbCase.y = dat[:,0]
    dbCase.yplus = dat[:,1]
    dbCase.uplus = dat[:,3]

    dbCase.h = 1.0
    dbCase.deltaplus = np.mean(dbCase.y[1:]/dbCase.yplus[...,1:])
    dbCase.utau = 1.0/dbCase.deltaplus
    dbCase.u = dbCase.uplus * dbCase.utau

    # Should do this properly later, checking what wall values actually are
    # and making sure it is compatible with viscosity law. But for present purposes just set to 1.0
    # as only care about ratios anyway
    dbCase.rho = dat[:,10]**2
    dbCase.rhow = 1.0
    dbCase.T = dat[:,11]
    dbCase.Tw = 1.0
    dbCase.mu = dat[:,12]
    dbCase.muw = 1.0

    save_case(dbCase, c[:-4] + ".dill")
    print(f"Saved case {cname} to database")
# %%
