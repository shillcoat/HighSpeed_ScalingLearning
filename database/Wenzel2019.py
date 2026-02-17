"""
Script to read in data files from Wenzel et al. (2019) and convert to database format

---------------------------------------------------------------------------------------------------

C. Wenzel, T. Gibis, M. Kloker, and U. Rist, “Self-similar compressible turbulent boundary layers with pressure gradients. 
Part 1. Direct numerical simulation and assessment of Morkovin’s hypothesis,” Journal of Fluid Mechanics, 
vol. 880, pp. 239–283, Dec. 2019, doi: 10.1017/jfm.2019.670.
"""

# %% Imports
import numpy as np
from glob import glob

from db_config import BL, save_case
from filepaths import Wenzel2019_path  # This file is not kept in repo
from prop_checks import check_edges

# %% Fetch and iterate through cases
cases = glob(Wenzel2019_path + "/data.npy/*.npy")
for c in cases:
    cname = c.split('/')[-1][:-4]
    dbCase = BL('si', incomp=0, chem=0, gamma=1.4, Pr=0.71, R=287.0, Bq=0.0)
    dat = np.load(c, allow_pickle=True)[()]

    dbCase.x = dat['geometry']['x']
    dbCase.y = dat['geometry']['y']

    dbCase.u = dat['2D_ReynoldsMean']['UMean']
    dbCase.v = dat['2D_ReynoldsMean']['VMean']
    dbCase.w = dat['2D_ReynoldsMean']['WMean']
    dbCase.a = dat['2D_ReynoldsMean']['aMean']
    dbCase.M = dat['2D_ReynoldsMean']['mMean']
    dbCase.rho = dat['2D_ReynoldsMean']['rhoMean']
    dbCase.P = dat['2D_ReynoldsMean']['pMean']
    dbCase.T = dat['2D_ReynoldsMean']['TMean']
    dbCase.mu = dat['2D_ReynoldsMean']['muMean']
    dbCase.nu = dat['2D_ReynoldsMean']['nuMean']

    dbCase.u_F = dat['2D_FavreMean']['UFavreMean']
    dbCase.v_F = dat['2D_FavreMean']['VFavreMean']
    dbCase.w_F = dat['2D_FavreMean']['WFavreMean']
    dbCase.rho_F = dat['2D_FavreMean']['rhoFavreMean']
    dbCase.T_F = dat['2D_FavreMean']['TFavreMean']
    dbCase.P_F = dat['2D_FavreMean']['pFavreMean']
    dbCase.mu_F = dat['2D_FavreMean']['muFavreMean']

    dbCase.upup = dat['2D_ReynoldsFlucts']['uI_uI']
    dbCase.vpvp = dat['2D_ReynoldsFlucts']['vI_vI']
    dbCase.wpwp = dat['2D_ReynoldsFlucts']['wI_wI']
    dbCase.rhoprhop = dat['2D_ReynoldsFlucts']['rI_rI']
    dbCase.TpTp = dat['2D_ReynoldsFlucts']['TI_TI']
    dbCase.PpPp = dat['2D_ReynoldsFlucts']['pI_pI']
    dbCase.upvp = dat['2D_ReynoldsFlucts']['uI_vI']
    dbCase.upwp = dat['2D_ReynoldsFlucts']['uI_wI']
    dbCase.vpwp = dat['2D_ReynoldsFlucts']['vI_wI']
    dbCase.upTp = dat['2D_ReynoldsFlucts']['uI_TI']
    dbCase.vpTp = dat['2D_ReynoldsFlucts']['vI_TI']
    dbCase.wpTp = dat['2D_ReynoldsFlucts']['wI_TI']

    dbCase.ruppupp = dat['2D_FavreFlucts']['r_uII_uII']
    dbCase.rvppvpp = dat['2D_FavreFlucts']['r_vII_vII']
    dbCase.rwppwpp = dat['2D_FavreFlucts']['r_wII_wII']
    dbCase.rrhopprhopp = dat['2D_FavreFlucts']['r_rII_rII']
    dbCase.rTppTpp = dat['2D_FavreFlucts']['r_TII_TII']
    dbCase.rPppPpp = dat['2D_FavreFlucts']['r_pII_pII']
    dbCase.ruppvpp = dat['2D_FavreFlucts']['r_uII_vII']
    dbCase.ruppwpp = dat['2D_FavreFlucts']['r_uII_wII']
    dbCase.rvppwpp = dat['2D_FavreFlucts']['r_vII_wII']
    dbCase.ruppTpp = dat['2D_FavreFlucts']['r_uII_TII']
    dbCase.rvppTpp = dat['2D_FavreFlucts']['r_vII_TII']
    dbCase.rwppTpp = dat['2D_FavreFlucts']['r_wII_TII']

    dbCase.tauw = dat['wallProperties']['tau_wall']
    dbCase.utau = dat['wallProperties']['utau']
    dbCase.Cf = dat['wallProperties']['cf']

    dbCase.delta99 = dat['boundaryLayerThicknesses']['delta99']
    dbCase.delta1 = dat['boundaryLayerThicknesses']['delta1']
    dbCase.delta2 = dat['boundaryLayerThicknesses']['theta']
    dbCase.H = dat['boundaryLayerThicknesses']['H12']
    dbCase.delta1k = dat['boundaryLayerThicknesses']['delta1Inc']

    # These are called "edge" values but when plotting the mean data they more
    # seem to correspond to the freestream values: possibly revisit this later
    dbCase.uinf = dat['boundaryLayerEdgeValues']['U_edge']
    dbCase.rhoinf = dat['boundaryLayerEdgeValues']['rho_edge']
    dbCase.Pinf = dat['boundaryLayerEdgeValues']['p_edge']
    dbCase.Tinf = dat['boundaryLayerEdgeValues']['T_edge']
    dbCase.Minf = dat['boundaryLayerEdgeValues']['m_edge']
    dbCase.muinf = dat['boundaryLayerEdgeValues']['nu_edge']*dbCase.rhoinf
    dbCase.uinf_F = dat['boundaryLayerEdgeValues']['UFavre_edge']
    dbCase.rhoinf_F = dat['boundaryLayerEdgeValues']['rhoFavre_edge']
    dbCase.Tinf_F = dat['boundaryLayerEdgeValues']['TFavre_edge']
    dbCase.Pinf_F = dat['boundaryLayerEdgeValues']['pFavre_edge']
    dbCase.muinf_F = dat['boundaryLayerEdgeValues']['muFavre_edge']

    dbCase.Redelta99 = dat['ReynoldsNumbers']['Re_delta']
    dbCase.Redelta1 = dat['ReynoldsNumbers']['Re_delta1']
    dbCase.Retheta = dat['ReynoldsNumbers']['Re_theta']
    dbCase.Retau = dat['ReynoldsNumbers']['Re_tau']

    # Some extra wall quantities
    dbCase.rhow = dbCase.rho[:,0]
    dbCase.muw = dbCase.mu[:,0]
    dbCase.Tw = dbCase.T[:,0]

    dbCase.Tr = dbCase.Tinf * (1 + 0.89*(dbCase.gamma-1.0)/2.0*dbCase.Minf**2)
    if cname[1] == 'A': dbCase.Bk = float(cname[-4:])
    elif cname[1] == 'F': dbCase.Bk = float(cname[-5:])
    else: dbCase.Bk = 0

    # Slice streamwise coordinate array to that which corresponds to available data
    Nx = 5000 if dbCase.Minf[0]>1.0 else 2500
    dbCase.x = dbCase.x[Nx-len(dbCase.u):-200]

    # Compute streamwise velocity profile in incompressible wall coordinates
    dbCase.uplus = dbCase.u/np.transpose([dbCase.utau])
    dbCase.deltaplus = dbCase.nu[:,0]/dbCase.utau
    dbCase.yplus = dbCase.y/np.transpose([dbCase.deltaplus])

    save_case(dbCase, Wenzel2019_path + f"/{cname}.dill")
    print(f"Saved case {cname} to database")
    check_edges(dbCase,title=cname)
# %%
