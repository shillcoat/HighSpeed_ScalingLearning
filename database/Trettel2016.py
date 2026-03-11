"""
Script to read in data files from Trettel and Larsson (2016) and convert to database format

A little quick and dirty for now as I'm limited on time

---------------------------------------------------------------------------------------------------

Add citation here
"""

# %% Imports
import numpy as np
import pandas as pd
from glob import glob

from db_config import Channel, save_case
from filepaths import Trettel2016_path  # This file is not kept in repo
from prop_checks import check_edges

import pdb

# %% Fetch and iterate through cases
cases = glob(Trettel2016_path + "/*_profiles.csv")

glbls = pd.read_csv(Trettel2016_path + "/globals.csv")
glbls.columns = glbls.columns.str.strip()  # Strip whitespace from column names

header_size = 1
for c in cases:
    cname = c.split('/')[-1][:-13]
    df = pd.read_csv(c); df.columns = df.columns.str.strip()
    cglbls = glbls[glbls["Originator's identifier"] == cname].iloc[0]

    params = {
        'gamma':      cglbls['"gamma"'],
        'R':          cglbls['"R"'],
        'Pr':         cglbls['"Pr"'],
        'Tw':         cglbls['"T_w"'],
        'Rebulk':     cglbls['"Re_bulk"'],
        'Mbulk':      cglbls['"Ma_bulk"'],
        'rhow':       cglbls['"rho_w"'],
        'muw':        cglbls['"mu_w"'],
        'tauw':       cglbls['"tau_w"'],
        'deltaplus':  cglbls['"l_nu"'],
        'utau':       cglbls['"u_tau"'],
        'Retau':      cglbls['"Re_tau"'],
        'Bq':         cglbls['"B_q"'],
        'qw':         cglbls['"q_w"'],
        'ue':         cglbls['"u_e"'],
        'rhoe':       cglbls['"rho_e"'],
        'Te':         cglbls['"T_e"'],
        'mue':        cglbls['"mu_e"'],
    }

    dbCase = Channel('nondim', incomp=0, chem=0, x=0, **params)
    
    # Should add viscosity power law later

    dat = np.genfromtxt(c, skip_header=header_size, delimiter=',')[:,:-1]
    dbCase.y = dat[:,0]
    dbCase.h = dbCase.y[-1]
    dbCase.yplus = dat[:,1]
    dbCase.u = dat[:,5]
    dbCase.u_F = dat[:,6]
    dbCase.uplus = dat[:,7]

    dbCase.rho = dat[:,11]
    dbCase.P = dat[:,12]
    dbCase.T = dat[:,13]
    dbCase.T_F = dat[:,14]
    dbCase.mu = dat[:,15]

    dbCase.rhoprhop = dat[:,23]
    dbCase.ruppupp = dbCase.rho * dat[:,16]
    dbCase.rvppvpp = dbCase.rho * dat[:,17]
    dbCase.rwppwpp = dbCase.rho * dat[:,18]
    dbCase.ruppvpp = dbCase.rho * dat[:,19]

    dbCase.PpPp = dat[:,24]
    dbCase.rTppTpp = dbCase.rho * dat[:,25]
    dbCase.rvppTpp = dbCase.rho * dat[:,26]

    save_case(dbCase, c[:-13] + ".dill")
    print(f"Saved case {cname} to database")