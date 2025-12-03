"""
Script to read in data files from Sillero et al. (2014) and convert to database format

---------------------------------------------------------------------------------------------------

J.A. Sillero, J. Jimenez, R.D. Moser
"Two-point statistics for turbulent boundary layers and channels 
at Reynolds numbers up to \delta^+\approx2000"
Phys. Fluids 26, 105109, 28 October 2014
https://pubs.aip.org/aip/pof/article/26/10/105109/103636/Two-point-statistics-for-turbulent-boundary-layers
"""

# %% Imports
import numpy as np
from glob import glob

from db_config import BL, save_case
from filepaths import Sillero2014_path  # This file is not kept in repo

# %% Fetch and iterate through cases
cases = glob(Sillero2014_path + "/profiles/*.prof")
header_size = 40
for c in cases:
    dbCase = BL('nondim', incomp=1, chem=0, Bq=0, Bk=0)

    # Read info from header
    with open(c,'r') as f:
        for i, line in enumerate(f):
            if i == 32:
                dbCase.utau = float(line[1:].split()[0].split("=")[1])
                dbCase.Retheta = float(line[1:].split()[2].split("=")[1])
                dbCase.Retau = float(line[1:].split()[4].split("=")[1])
                dbCase.Redelta2 = dbCase.Retheta  # These are the same for incompressible flow
            if i == 33:
                dbCase.delta99 = float(line[1:].split()[0].split("=")[1])
                dbCase.delta1 = float(line[1:].split()[2].split("=")[1])
                dbCase.delta2 = float(line[1:].split()[4].split("=")[1])
                dbCase.H = dbCase.delta1 / dbCase.delta2
                break

    dat = np.loadtxt(c, skiprows=header_size)

    dbCase.y = dbCase.delta99 * dat[:,0]
    dbCase.yplus = dat[:,1]

    # Mean quantities
    dbCase.uplus = dat[:,8]
    dbCase.u = dbCase.uplus * dbCase.utau
    dbCase.uinf = dbCase.u[-1]
    dbCase.nu = dbCase.uinf * dbCase.delta2 / dbCase.Retheta
    dbCase.Cf = 2/dbCase.uplus[-1]**2

    dbCase.upup = np.square(dbCase.utau * dat[:,2])
    dbCase.vpvp = np.square(dbCase.utau * dat[:,3])
    dbCase.wpwp = np.square(dbCase.utau * dat[:,4])
    dbCase.upvp = np.square(dbCase.utau * dat[:,5])
    dbCase.upwp = np.square(dbCase.utau * dat[:,6])
    dbCase.vpwp = np.square(dbCase.utau * dat[:,7])

    cname = c.split('/')[-1][:-5]
    i = c[::-1].index('/') + 1
    save_case(dbCase, c[:-i-8] + f"{cname}.dill")
# %%
