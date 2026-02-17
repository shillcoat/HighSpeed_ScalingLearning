"""
Script to read in data files from Lee & Moser (2015) and convert to database format

---------------------------------------------------------------------------------------------------

Myoungkyu Lee and Robert D. Moser,  
Direct numerical simulation of turbulent channel flow up to Re_tau = 5200, 
2015, Journal of Fluid Mechanics, vol. 774, pp. 395-415
http://journals.cambridge.org/article_S0022112015002682
"""

# %% Imports
import numpy as np
from glob import glob

from db_config import Channel, save_case
from filepaths import Lee2015_path  # This file is not kept in repo

# %% Fetch and iterate through cases
cases = glob(Lee2015_path + "/LM*/*mean_prof.dat")
header_size = 72
for c in cases:
    dbCase = Channel('nondim', incomp=1, chem=0)

    # Read info from header
    with open(c,'r') as f:
        for i, line in enumerate(f):
            if i == 38:
                nu = float(line[1:].split()[4])
            if i == 39:
                dbCase.h = float(line[1:].split()[5])
            if i == 41:
                dbCase.utau = float(line[1:].split()[4])
            if i == 42:
                dbCase.Retau = float(line[1:].split()[3])
                break

    dat = np.loadtxt(c, skiprows=header_size)

    dbCase.y = dbCase.h * dat[:,0]
    dbCase.nu = np.ones_like(dbCase.y) * nu
    dbCase.yplus = dat[:,1]
    dbCase.uplus = dat[:,2]

    i = c[::-1].index('/') + 1
    save_case(dbCase, c[:-i] + ".dill")
# %%
