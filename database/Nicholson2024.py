"""
Script to read in data files from Nicholson et al. (2024) and convert to database format

The variables available online are somewhat limited, but Prof. Duan did offer to provide
additional data if needed so I may reach out if this is too limiting. Start with this and see
if I get anything promising out of it that warrants going through it more rigorously

A lot of uncertainty on how to define things here so can use in database:
- How to handle presence of shocks?
- BL thickness/free-stream values vs BL edge values -> No longer functionally equivalent (ex. Retaustar)

---------------------------------------------------------------------------------------------------

[1] G. L. Nicholson, L. Duan, and N. J. Bisek, “Direct Numerical Simulation Database of High-Speed Flow over 
Parameterized Curved Walls,” AIAA Journal, vol. 62, no. 6, pp. 2095–2118, June 2024, doi: 10.2514/1.J063456.

"""

# %% Imports
import numpy as np
from glob import glob

from db_config import BL, save_case
from filepaths import Nicholson2019_path  # This file is not kept in repo

# %% Fetch and iterate through cases
cases = glob(Nicholson2019_path + "/*_Stat.dat")
# header_size = 123
K = 320  # Number of data points/zone
H = 19 / 1e3  # Total difference in height of wall due to curvature (from paper)
for c in cases:
    dbCase = BL('si', incomp=0, chem=0, gamma=1.4, Pr=0.71, R=287.0)
    dbCase.mu_law = lambda T: 1.458e-6*T**(3.0/2.0)/(T+110.4)

    # Need to be careful, each file contains multiple stations... how to read these?
    # In paper subscript 1 denotes properties before the wall curvature

    # General properties at stations U1 and U2 upstream of curvature are given and temporarily
    # stored in dictionaries as may be useful later in script
    U1, U2 = {}, {}

    with open(c,'r') as f:
        for i, line in enumerate(f):
            if i == 17:
                params = line[1:].split()
                dbCase.Tw = float(params[4])
                Tr1 = dbCase.Tw / float(params[5])

                # Inlet BL thickness
                deltai = float(params[6]) / 1e3

                # Pre-wall curvature free-stream values
                Minf1 = float(params[0])
                Uinf1 = float(params[1])
                rhoinf1 = float(params[2])
                Tinf1 = float(params[3])
            if i == 21:
                params = line[1:].split()
                U1['x'] = float(params[1])
                U1['Retheta'] = float(params[2])
                U1['Retau'] = float(params[3])
                U1['Redelta2'] = float(params[4])
                U1['Retaustar'] = float(params[5])
                U1['delta2'] = float(params[6]) / 1e3
                U1['H12'] = float(params[7])
                U1['delta'] = float(params[8]) / 1e3
                U1['deltaplus'] = float(params[9]) / 1e6
                U1['utau'] = float(params[10])
            if i == 22:
                params = line[1:].split()
                U2['x'] = float(params[1])
                U2['Retheta'] = float(params[2])
                U2['Retau'] = float(params[3])
                U2['Redelta2'] = float(params[4])
                U2['Retaustar'] = float(params[5])
                U2['delta2'] = float(params[6]) / 1e3
                U2['H12'] = float(params[7])
                U2['delta'] = float(params[8]) / 1e3
                U2['deltaplus'] = float(params[9]) / 1e6
                U2['utau'] = float(params[10])
            if "DT=(DOUBLE" in line:
                header_size = i+1
                break

    # Read data from each zone and put it all together
    datU2 = np.loadtxt(c, skiprows=header_size, max_rows=K)
    datL1 = np.loadtxt(c, skiprows=header_size+K+5, max_rows=K)
    datL2 = np.loadtxt(c, skiprows=header_size+2*(K+5), max_rows=K)
    datL3 = np.loadtxt(c, skiprows=header_size+3*(K+5), max_rows=K)
    datL4 = np.loadtxt(c, skiprows=header_size+4*(K+5), max_rows=K)
    if '1p00' in c:
        datE = np.loadtxt(c, skiprows=header_size+5*(K+5), max_rows=K)
        dat = np.array([datU2, datE, datL1, datL2, datL3, datL4])
    else:
        dat = np.array([datU2, datL1, datL2, datL3, datL4])
    del datU2, datL1, datL2, datL3, datL4  # Clean up a little
    nx, ny = dat.shape[:2]

    # Stuff gets weird because of curved walls. For now I'm going to choose y in the local frame
    # (normal to wall at that point) as velocity data is in this reference frame. The x coordinate
    # will be x_g at the wall (where the wall-normal line shoots off from)
    dbCase.x = dat[:,0,0]
    dxg = dat[:,:,0].transpose() - dbCase.x
    dyg = dat[:,:,1].transpose() - dat[:,0,1]
    dbCase.y = np.sqrt(np.square(dxg)+np.square(dyg)).transpose()
    
    # Go through rest of data available data is pretty limited unfortunately
    dbCase.yplus = dat[:,:,3]
    dbCase.ystar = dat[:,:,4]
    dbCase.u_F = dat[:,:,5]*Uinf1
    dbCase.v_F = dat[:,:,6]*Uinf1
    dbCase.rho = dat[:,:,7]*rhoinf1  # I believe this is Reynolds average density but I'm not 100%
    dbCase.rhow = dbCase.rho[:,0]
    dbCase.T_F = dat[:,:,8]*Tinf1
    dbCase.mu_F = dbCase.mu_law(dbCase.T_F)
    dbCase.muw = dbCase.mu_F[:,0]
    dbCase.ruppupp = dat[:,:,9]*rhoinf1*Uinf1**2
    dbCase.rwppwpp = dat[:,:,10]*rhoinf1*Uinf1**2
    dbCase.rvppvpp = dat[:,:,11]*rhoinf1*Uinf1**2
    dbCase.ruppvpp = dat[:,:,12]*rhoinf1*Uinf1**2
    dbCase.ruppTpp = dat[:,:,13]*rhoinf1*Uinf1*Tinf1
    dbCase.rvppTpp = dat[:,:,13]*rhoinf1*Uinf1*Tinf1

    # Somehow this dataset does not provide pressure? Recover from perfect gas EOS but may be issues
    # with averaging
    dbCase.P_F = dbCase.rho*dbCase.R*dbCase.T_F

    # I'll get a speed of sound/Mach number but unsure how sound they are as it used Favre-averaged quantities
    # Better than nothing I guess
    dbCase.a = np.sqrt(dbCase.gamma*dbCase.R*dbCase.T_F)
    dbCase.M = dbCase.u_F/dbCase.a

    # Get some wall quantities from the corresponding wallParams.dat file
    cw = c[:-8] + 'wallParams.dat'
    header_size = 93
    datw = np.loadtxt(cw, skiprows=header_size)
    cf, ch, cp = np.array([]), np.array([]), np.array([])
    for xloc in dbCase.x:
        ix = np.argmin(datw[:,0]-xloc)
        cf = np.append(cf,datw[ix,-3])
        ch = np.append(ch,datw[ix,-2])
        cp = np.append(cp,datw[ix,-1])
    dbCase.Cf = cf
    dbCase.tauw = cf*rhoinf1*Uinf1**2/2.0
    dbCase.utau = np.sqrt(dbCase.tauw/dbCase.rhow)
    dbCase.Mtau = dbCase.utau/np.sqrt(dbCase.gamma*dbCase.R*dbCase.Tw)
    cap_p = dbCase.R*dbCase.gamma/(dbCase.gamma-1.0)
    dbCase.qw = ch*rhoinf1*Uinf1*cap_p*(Tr1-dbCase.Tw)
    dbCase.Bq = dbCase.qw/(dbCase.rhow*cap_p*dbCase.utau*dbCase.Tw)

    # Get friction scales/parameters (will have to substitute in Favre-averaged data where normally use
    # Reynolds-averaged data instead)
    dbCase.deltaplus = dbCase.muw/dbCase.rhow/dbCase.utau
    dbCase.deltastar = dbCase.mu_F/np.transpose(np.sqrt(dbCase.rhow*np.transpose(dbCase.rho))*dbCase.utau)
    dbCase.yplus = dbCase.y/np.transpose([dbCase.deltaplus])
    dbCase.ystar = dbCase.y/dbCase.deltastar
    dbCase.uplus = dbCase.u_F/np.transpose([dbCase.utau])

    # Get some free-stream quantities (just take furthest available point from wall as otherwise idk what to do)
    dbCase.Minf = dbCase.M[:,-1]
    dbCase.ainf = dbCase.a[:,-1]
    dbCase.uinf_F = dbCase.u_F[:,-1]
    dbCase.Tinf_F = dbCase.T_F[:,-1]
    dbCase.Pinf_F = dbCase.P_F[:,-1]
    dbCase.muinf_F = dbCase.mu_F[:,-1]
    dbCase.rhoinf = dbCase.rho[:,-1]

    dbCase.Tr = dbCase.Tinf_F * (1 + 0.89*(dbCase.gamma-1.0)/2.0*dbCase.Minf**2)

    # Get visual thickness (again using Favre average cause that's all I have)
    id99 = np.zeros([nx],dtype=np.int64)
    for j in range(ny):
        id99 += dbCase.u_F[:,j] < 0.99*dbCase.uinf_F
    id99 = np.transpose(np.array(list(zip(range(nx),id99))))
    dbCase.delta99 = dbCase.y[*id99]
    # Friction Reynolds numbers
    dbCase.Retau = dbCase.delta99 / dbCase.deltaplus
    # dbCase.Retaustar = dbCase.delta99 / dbCase.deltastar[*id99]
    # Above doesn't agree well with values reported in paper at first station.
    # Below agrees better, emphasizes that need to carefully consider difference
    # between BL edge and free-stream values
    dbCase.Retaustar = dbCase.delta99 / dbCase.deltastar[:,-1]

    # Other BL thicknesses and corresponding parameters
    I1 = 1-(dbCase.rho*dbCase.u_F)/np.transpose([(dbCase.rhoinf*dbCase.uinf_F)])
    I1k = 1-dbCase.u_F/np.transpose([dbCase.uinf_F])
    I2 = (dbCase.rho*dbCase.u_F)/np.transpose([(dbCase.rhoinf*dbCase.uinf_F)])*(1-dbCase.u_F/np.transpose([dbCase.uinf_F]))
    dbCase.delta1 = np.zeros([nx])
    dbCase.delta1k = np.zeros([nx])
    dbCase.delta2 = np.zeros([nx])
    dbCase.H = np.zeros([nx])
    Pe = np.zeros([nx])
    for xi in range(nx):
        y = dbCase.y[xi,:id99[1,xi]]
        dbCase.delta1[xi] = np.trapezoid(I1[xi,:id99[1,xi]],y)
        dbCase.delta1k[xi] = np.trapezoid(I1k[xi,:id99[1,xi]],y)
        dbCase.delta2[xi] = np.trapezoid(I2[xi,:id99[1,xi]],y)
        Pe[xi] = dbCase.P_F[xi,id99[1,xi]]
    dbCase.H = dbCase.delta1/dbCase.delta2
    dbCase.Redelta99 = dbCase.rhoinf*dbCase.uinf_F*dbCase.delta99/dbCase.muinf_F
    dbCase.Redelta1 = dbCase.rhoinf*dbCase.uinf_F*dbCase.delta1/dbCase.muinf_F
    dbCase.Redelta2 = dbCase.rhoinf*dbCase.uinf_F*dbCase.delta2/dbCase.muw
    dbCase.Retheta = dbCase.rhoinf*dbCase.uinf_F*dbCase.delta2/dbCase.muinf_F
    # Unfortunately due to coarse grain of stations I can only get very coarse-grained outer pressure gradient
    dbCase.Bk = dbCase.delta1k/dbCase.tauw*np.gradient(Pe,dbCase.x)
    # dbCase.Bk = dbCase.delta1k/dbCase.tauw*np.gradient(dbCase.Pinf_F,dbCase.x)

    cname = c.split('/')[-1][:-9]
    save_case(dbCase, Nicholson2019_path + f"/{cname}.dill")
# %%
