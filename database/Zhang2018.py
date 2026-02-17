"""
Script to read in data files from Zhang et al. (2018) and convert to database format
They don't appear to provide Favre fluctuating quantities: they do have Reynolds stress anisotropies
however would need TKE to reconstruct Favre fluctuations from these

---------------------------------------------------------------------------------------------------

Zhang, C., Duan, L., and Choudhari, M., "Direct Numerical Simulation Database for Supersonic 
and Hypersonic Turbulent Boundary Layers," published in AIAA Journal, 2018. DOI: 10.2514/1.J057296
https://arc.aiaa.org/doi/pdf/10.2514/1.J057296
"""

# %% Imports
import numpy as np
from glob import glob

from db_config import BL, save_case
from filepaths import Zhang2018_path  # This file is not kept in repo
from prop_checks import check_edges

# %% Fetch and iterate through cases
cases = glob(Zhang2018_path + "/*_Stat.dat")
header_size = 142
for c in cases:
    cname = c.split('/')[-1][:-9]
    dbCase = BL('si', incomp=0, chem=0, gamma=1.4, Pr=0.71, Bk=0)
    if "M8Tw048" in c:  # Case with nitrogen as working fluid
         dbCase.R = 297.0
         dbCase.mu_law = lambda T: 1.418e-6*T**(3.0/2.0)/(T+116.4*10**(-5.0/T))
    else:  # Working fluid is air for all other cases
         dbCase.R = 287.0
         dbCase.mu_law = lambda T: 1.458e-6*T**(3.0/2.0)/(T+110.4)

    # Read info from header
    with open(c,'r') as f:
        for i, line in enumerate(f):
            if i == 26:
                params = line[1:].split()
                dbCase.Minf = float(params[0])
                dbCase.uinf = float(params[1])
                dbCase.ainf = dbCase.uinf/dbCase.Minf
                dbCase.rhoinf = float(params[2])
                dbCase.Tinf = float(params[3])
                dbCase.Pinf = dbCase.rhoinf*dbCase.R*dbCase.Tinf
                dbCase.Tw = float(params[4])
                TwTr = float(params[5])
                dbCase.Tr = dbCase.Tw / TwTr
                delta_in = float(params[6])/1e3
            if i == 28:
                params = line[1:].split()
                dbCase.x = delta_in*float(params[0])
                dbCase.Retheta = float(params[1])
                dbCase.Retau = float(params[2])
                dbCase.Redelta2 = float(params[3])
                dbCase.Retaustar = float(params[4])
                dbCase.delta2 = float(params[5])/1e3
                dbCase.H = float(params[6])
                dbCase.delta1 = dbCase.H * dbCase.delta2
                dbCase.delta99 = float(params[7])/1e3
                dbCase.deltaplus = float(params[8])/1e3
                dbCase.utau = float(params[9])
                dbCase.Bq = -float(params[10])
                dbCase.Mtau = float(params[11])
                break

    # Read data
    dat = np.loadtxt(c, skiprows=header_size + (2 if "M14Tw018" in c else 0))
    
    dbCase.y = dat[:,0]
    dbCase.yplus = dat[:,2]
    dbCase.ystar = dat[:,3]

    # Mean quantities
    dbCase.u = dbCase.uinf * dat[:,4]
    dbCase.uplus = dbCase.u / dbCase.utau
    dbCase.P = dbCase.Pinf * dat[:,5]
    dbCase.T = dbCase.Tinf * dat[:,6]
    dbCase.rhow = dbCase.P[0]/dbCase.R/dbCase.Tw
    dbCase.muw = dbCase.mu_law(dbCase.Tw)
    dbCase.rho = dbCase.rhow * dat[:,7]
    dbCase.mu = dbCase.mu_law(dbCase.T)
    dbCase.tauw = dbCase.utau**2*dbCase.rho[0]
    dbCase.a = np.sqrt(dbCase.gamma*dbCase.R*dbCase.T)
    dbCase.M = dbCase.u/dbCase.a

    # Fluctuating quantities
    dbCase.upup = np.square(dbCase.utau * dat[:,8])
    dbCase.vpvp = np.square(dbCase.utau * dat[:,10])
    dbCase.wpwp = np.square(dbCase.utau * dat[:,9])
    dbCase.PpPp = np.square(dbCase.Pinf * dat[:,11])
    dbCase.TpTp = np.square(dbCase.Tinf * dat[:,12])
    dbCase.rhoprhop = np.square(dbCase.rhoinf * dat[:,13])
    dbCase.MpMp = np.square(dat[:,14])
    dbCase.upvp = dbCase.utau**2 * dat[:,15]

    # Other
    dbCase.Prt = dat[:,-2]
    dbCase.Cf = 2*dbCase.tauw/(dbCase.rhoinf + dbCase.uinf**2)

    save_case(dbCase, c[:-9] + ".dill")
    print(f"Saved case {cname} to database")
    check_edges(dbCase,title=cname)
# %%
