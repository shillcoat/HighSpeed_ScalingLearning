"""
Script to read in data files from Larsson group boundary layer data and convert to database format
Files are provided with inputs and quantities along the wall. Wall-normal profiles are only
provided at a single streamwise location, so not used here.

---------------------------------------------------------------------------------------------------

https://larsson.umd.edu/data/
"""

# %% Imports
import numpy as np
from glob import glob

from db_config import BL, save_case
from filepaths import LarssonGroupBL_path  # This file is not kept in repo

# %% Fetch and iterate through cases
cases = glob(LarssonGroupBL_path + "/*/*.streamwise")
for c in cases:
    casename = c.split('/')[-1][:-11]
    TwTr = float(casename.split('r')[1].split('R')[0])/10
    
    with open(c[:-11]+'.input','r') as f:
        lines = f.readlines(); i = 0
        Tw = float(lines[4].split()[1])
        while 'GAS' not in lines[i]: i+=1
        R = float(lines[i+1].split()[1])
        muref, Tref = float(lines[i+2].split()[0]), float(lines[i+2].split()[1])
    dbCase = BL('nondim', incomp=0, chem=0, gamma=1.4, Pr=0.7, Bk=0, R=R, Tw=Tw)
    
    dbCase.mu_law = lambda T: muref * (T/Tref)**0.75  # Exponent of power law not specified but this is common
    cp = dbCase.R * (dbCase.gamma/(dbCase.gamma-1))
    dbCase.Minf = float(casename.split('T')[0][1:])
    dbCase.Tr = Tw / TwTr
    dbCase.Tinf = dbCase.Tr / (1 + 0.89*(dbCase.gamma-1)/2 * dbCase.Minf**2)
    dbCase.muw = dbCase.mu_law(Tw)
    dbCase.muinf = dbCase.mu_law(dbCase.Tinf)

    dat = np.loadtxt(c, comments='%')
    
    dbCase.x = dat[:,0]
    dbCase.tauw = dat[:,1]
    dbCase.qw = dat[:,2]
    dbCase.delta99 = dat[:,3]
    dbCase.delta2 = dat[:,4]
    dbCase.delta2k = dat[:,5]
    dbCase.delta1 = dat[:,6]
    dbCase.delta1k = dat[:,7]
    dbCase.Redelta99 = dat[:,12]
    dbCase.Retheta = dat[:,13]
    dbCase.Redelta2 = dat[:,14]
    dbCase.Retau = dat[:,15]
    dbCase.Retaustar = dat[:,16]
    dbCase.Mtau = dat[:,18]
    dbCase.Bq = dat[:,19]
    dbCase.Cf = dat[:,20]

    dbCase.uinf = 1.0 # From non-dimensionalization
    dbCase.rhoinf = 2*dbCase.tauw/dbCase.Cf
    dbCase.rhow = (dbCase.qw / dbCase.Bq / cp / Tw)**2 / dbCase.tauw
    dbCase.utau = np.sqrt(dbCase.tauw / dbCase.rhow)

    save_case(dbCase, LarssonGroupBL_path + f"/{casename}.dill")

# %%
