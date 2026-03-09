import dill
import numpy as np
from warnings import warn
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter

import pdb

__all__ = ['load_case',
           'Case',
           'BL',
           'Channel',
           'Pipe',
           'Duct']

def load_case(fpath):
    with open(fpath, 'rb') as f:
        c = dill.load(f)
    t = __all__[[s.lower() for s in __all__].index(c['case_type'])]
    c.pop('case_type')
    return globals()[t](**c)

def save_case(case, fpath):
    # Save all case info as a dict so independent of class
    p = vars(case)
    poplist = []
    for key in p.keys():  # Pop all "private" members
        if key[0] == '_': poplist.append(key)
    for k in poplist: p.pop(k)
    with open(fpath, 'wb') as f:
        dill.dump(p,f)


class Case:
    """
    Class to provide uniform structure for all cases contained in database.

    Designed so that no new members may be added after initialization (to ensure 
    meets database format)
    """

    field_params = {
        # Reynolds-averaged mean quantities
        "u": "Reynolds-averaged mean streamwise velocity",
        "v": "Reynolds-averaged mean wall-normal velocity",
        "w": "Reynolds-averaged mean spanwise velocity",
        "a": "Reynolds-averaged mean speed of sound",
        "M": "Reynolds-averaged mean Mach number",
        "Mt": "Reynolds-averaged mean turbulent Mach number",
        "rho": "Reynolds-averaged mean density",
        "P": "Reynolds-averaged mean pressure",
        "T": "Reynolds-averaged mean temperature",
        "mu": "Reynolds-averaged mean dynamic viscosity",
        "nu": "Reynolds-averaged mean kinematic viscosity",

        # Favre-averaged mean quantities
        "u_F": "Favre-averaged mean streamwise velocity",
        "v_F": "Favre-averaged mean wall-normal velocity",
        "w_F": "Favre-averaged mean spanwise velocity",
        "rho_F": "Favre-averaged mean density",
        "P_F": "Favre-averaged mean pressure",
        "T_F": "Favre-averaged mean temperature",
        "mu_F": "Favre-averaged mean dynamic viscosity",

        # Reynolds-averaged fluctuating quantities
        "upup": "<u'u'>",
        "vpvp": "<v'v'>",
        "wpwp": "<w'w'>",
        "upvp": "<u'v'>",
        "upwp": "<u'w'>",
        "vpwp": "<v'w'>",
        "rhoprhop": "<rho'rho'>",
        "PpPp": "<P'P'>",
        "TpTp": "<T'T'>",
        "upTp": "<u'T'>",
        "vpTp": "<v'T'>",
        "wpTp": "<w'T'>",
        "MpMp": "<M'M'>",
        "k": "Turbulent kinetic energy",
        
        # Favre-averaged fluctuating quantities
        "ruppupp": "<rho u''u''>",
        "rvppvpp": "<rho v''v''>",
        "rwppwpp": "<rho w''w''>",
        "ruppvpp": "<rho u''v''>",
        "ruppwpp": "<rho u''w''>",
        "rvppwpp": "<rho v''w''>",
        "rrhopprhopp": "<rho rho''rho''>",
        "rPppPpp": "<rho P''P''>",
        "rTppTpp": "<rho T''T''>",
        "ruppTpp": "<rho u''T''>",
        "rvppTpp": "<rho v''T''>",
        "rwppTpp": "<rho w''T''>",

        # Scaling
        "deltastar": "Semi-local length scale",
        "yplus": "Wall-normal coordinate in friction units",
        "ystar": "Wall-normal coordinate in semi-local units",
        "uplus": "Streamwise velocity in frction units",

    }
    oneD_params = {
        # Geometry
        "x": "Raw streamwise coordinate",
        "y": "Raw wall-normal coordinate",
        "z": "Raw spanwise coordinate",

        # Flags/bookkeeping
        "incomp": "Flag for incompressible cases",
        "chem": "Flag for chemically-reacting cases",
        "mu_law": "Function handle for viscosity law",

        # Working fluid properties
        "R": "Gas constant",
        "gamma": "Specific heat ratio",

        # Wall quantities
        "rhow": "Wall density",
        "Tw": "Wall temperature",
        "muw": "Wall dynamic viscosity",
        "tauw": "Wall shear stress",
        "qw": "Wall heat flux",
        "utau": "Friction velocity",
        "Mtau": "Friction Mach number",
        "Cf": "Skin-friction coefficient",
        "Bq": "Dimensionaless wall heat-transfer rate",

        # Edge quantities
        "ue": "Reynolds-averaged mean edge velocity",
        "ae": "Reynolds-averaged mean edge speed of sound",
        "Me": "Reynolds-averaged mean edge Mach number",
        "rhoe": "Reynolds-averaged mean edge density",
        "Pe": "Reynolds-averaged mean edge pressure",
        "Te": "Reynolds-averaged mean edge temperature",
        "mue": "Reynolds-averaged mean edge dynamic viscosity",
        "ue_F": "Favre-averaged mean edge velocity",
        "rhoe_F": "Favre-averaged mean edge density",
        "Pe_F": "Favre-averaged mean edge pressure",
        "Te_F": "Favre-averaged mean edge temperature",
        "mue_F": "Favre-averaged mean edge dynamic viscosity",

        # Scaling
        "deltaplus": "Friction length scale",

        # Non-dimensional numbers
        "Pr": "Molecular Prandtl number",
        "Prt": "Reynolds Turbulent Prandtl number",
        "Prt_F": "Favre Turbulent Prandtl number",
        "Retau": "Friction Reynolds number",
        "Retaustar": "Semi-local friction Reynolds number",
    }
    
    # List of all parameters with description
    gen_params = oneD_params | field_params
    __frozen = 0

    @staticmethod
    def whatis(p):
        # Return description for parameter p
        if p in Case.gen_params.keys():
            print(Case.gen_params[p])
        else:
            warn("%r is not a valid case parameter" % p)

    def __init__(self, case_type:str, units:str, **params):
        self.case_type = str.lower(case_type)
        self.units = str.lower(units)
        
        # Initialize all the members listed in the params list
        for key in Case.gen_params.keys():
            setattr(self, key, None)

        # Read in provided parameters and assign those that are valid
        for key, val in params.items():
            if key in Case.gen_params.keys():
                setattr(self, key, val)
            else:
                warn("%r is not a valid parameter and will be ignored" % key)

        self._freeze()

    def __setattr__(self, key, value):
        # Override setattr to support freezing of attributes
        # Ensure that all attributes are stored as numpy arrays for consistency, 
        # but allow scalars to be input for convenience
        if self.__frozen and not hasattr(self, key):
            raise AttributeError("%r is frozen: cannot add additional members after " \
            "initialization" % self)
        if value is not None:  # Some input cleaning/pre-processing for consistency
            scalar_check = np.isscalar(value) and not isinstance(value, str|bool)
            try: # Handling of numpy 0d arrays (super annoying that these exist...)
                if not value.ndim and isinstance(value, np.ndarray): value = value[np.newaxis]
            except AttributeError:
                pass
            if key in Case.field_params.keys() and value.ndim < 2:
                # Ensure the field parameters are always at least 2D for consistency 
                # (even if only one point in x direction)
                value = value[np.newaxis, :]
            super().__setattr__(key, np.array([value]) if scalar_check else value)
        else: super().__setattr__(key, value)

    def _freeze(self):
        # Freeze object so can't add new attributes
        self.__frozen = 1

    def _unfreeze(self):
        # Unfreeze object so can add new attributes: should never need to use!
        self.__frozen = 0

    def hasdata(self, p=None):
        # If no parameters p given, returns a list of parameters containing data.
        # If a list of parameters p is given, returns a corresponding list of booleans
        # for if they contain data
        r = []
        if p is None:
            for param in Case.gen_params.keys():
                if getattr(self,param,None) is not None: r.append(param)
            return r
        if isinstance(p, str): return getattr(self, p, None) is not None
        for param in p:
            r.append(getattr(self, param, None) is not None)
        return r

    def vel_transform(self,f=None,g=None,fw=1,gw=1,label:str=""):
        # Perform velocity transformation with given transformation kernels f and g
        # fw and gw are wall values of kernel that will be prepended if y=0 is not included in dataset
        # If a label is provided, the transformed velocity and coordinate will be saved to u{label} and y{label} attributes
        ndim = len(self.u.shape)
        prep = True if np.all(self.y[...,0]>1e-10) else False
        if label:
            self._unfreeze()
            rhop = self.rho/np.transpose([self.rhow])
            mup = self.mu/np.transpose([self.muw])

            # Check if label corresponds to one of known transformations
            if label.upper() == "VD":
                # Van Driest
                f = np.ones(self.u.shape)
                g = np.sqrt(rhop)
            elif label.upper() == "TL":
                # Trettel & Larsson
                if len(self.y.shape) == 1:
                    f = np.gradient(self.y*np.sqrt(rhop)/mup,self.y,axis=-1,edge_order=2)
                else:
                    f = np.zeros(self.u.shape)
                    for xi in range(self.u.shape[0]):
                        f[xi] = np.gradient(self.y[xi]*np.sqrt(rhop[xi])/mup[xi],self.y[xi],axis=-1,edge_order=2)
                g = mup * f
            elif label.upper() == "V":
                # Volpiani et al.
                f = np.sqrt(rhop)/mup**(3.0/2.0)
                g = np.sqrt(rhop)/np.sqrt(mup)
            elif label.upper() == "GFM":
                # Griffin, Fu, Moin: use constant stress assumption as shown to make negligible difference
                if not all(self.hasdata(['uTL', 'yTL'])):
                    _ = self.vel_transform(label="TL")
                if len(self.y.shape) == 1:
                    f = np.gradient(self.y*np.sqrt(rhop)/mup,self.y,axis=-1,edge_order=2)
                else:
                    f = np.zeros(self.u.shape)
                    for xi in range(self.u.shape[0]):
                        f[xi] = np.gradient(self.y[xi]*np.sqrt(rhop[xi])/mup[xi],self.y[xi],axis=-1,edge_order=2)
                if ndim > 1:
                    Seq, Stl = np.zeros(self.u.shape), np.zeros(self.u.shape)
                    # Unfortuntely np.gradient doesn't support multi-dim spacing specs so have to do this to support 2D data
                    for i in range(len(self.x)):
                        Seq[i,:] = np.gradient(self.uplus[i,:],self.yplusTL[i,:],axis=-1,edge_order=2)/mup[i,:]
                        Stl[i,:] = np.gradient(self.uplusTL[i,:],self.yplusTL[i,:],axis=-1,edge_order=2)
                else:
                    Seq = np.gradient(self.uplus,self.yplusTL,axis=-1,edge_order=2)/mup
                    Stl = np.gradient(self.uplusTL,self.yplusTL,axis=-1,edge_order=2)
                # tauv = self.mu*np.gradient(self.u,self.y)
                # tauR = -self.ruppvpp
                # tp = (tauv+tauR)/self.tauw
                tp = 1
                g = tp/mup/(tp+Seq-Stl)
            elif label.upper() == "H":
                # Hasan et al. -> Might be interesting to do scaling analysis for eddy viscosity rather than just fit
                pass

        if f is None or g is None:
            raise ValueError("Transformation is not known and no kernels were provided")

        if prep:
            f = np.insert(f,0,fw,axis=1 if ndim>1 else 0)
            g = np.insert(g,0,gw,axis=1 if ndim>1 else 0)    

        # Perform the numerical integration using given kernels
        yscaled, uscaled = np.zeros(self.u.shape), np.zeros(self.u.shape)
        sj = np.s_[:,0] if ndim>1 else np.s_[0]
        sjp1 = np.s_[:,1] if ndim>1 else np.s_[1]
        uscaled[sj] = (g[sj]+g[sjp1])/2.0 * self.u[sj]
        yscaled[sj] = (f[sj]+f[sjp1])/2.0 * self.y[...,0]

        if prep:
            # Cut out prepended wall entry to simplify slicing
            g = g[1:] if ndim==1 else g[:,1:]
            f = f[1:] if ndim==1 else f[:,1:]

        for j in range(1, self.y.shape[-1]):
            uscaled[sjp1] = uscaled[sj] + (g[sj]+g[sjp1])/2.0*(self.u[sjp1]-self.u[sj])
            yscaled[sjp1] = yscaled[sj] + (f[sj]+f[sjp1])/2.0*(self.y[...,j]-self.y[...,j-1])
            sj = sjp1
            sjp1 = np.s_[:,j+1] if ndim>1 else np.s_[j+1]

        if label is not None:
            setattr(self, f'y{label}', yscaled)
            setattr(self, f'u{label}', uscaled)
            setattr(self, f'yplus{label}', yscaled/np.transpose([self.deltaplus]))
            setattr(self, f'uplus{label}', uscaled/np.transpose([self.utau]))
            self._freeze()
        return yscaled, uscaled


class BL(Case):
    bl_params = {
        # BL Geometry
        'delta99': 'Visual thickness',
        'delta1': 'Displacement thickness',
        'delta1k': 'Kinematic displacement thickness',
        'delta2': 'Momentum thickness (often written as theta)',
        'delta2k': 'Kinematic momentum thickness',
        'H': 'Shape factor',

        # Free-stream quantities
        "uinf": "Reynolds-averaged mean free-stream velocity",
        "ainf": "Reynolds-averaged mean free-stream speed of sound",
        "Minf": "Reynolds-averaged mean free-stream Mach number",
        "rhoinf": "Reynolds-averaged mean free-stream density",
        "Pinf": "Reynolds-averaged mean free-stream pressure",
        "Tinf": "Reynolds-averaged mean free-stream temperature",
        "muinf": "Reynolds-averaged mean free-stream dynamic viscosity",
        "uinf_F": "Favre-averaged mean free-stream velocity",
        "rhoinf_F": "Favre-averaged mean free-stream density",
        "Pinf_F": "Favre-averaged mean free-stream pressure",
        "Tinf_F": "Favre-averaged mean free-stream temperature",
        "muinf_F": "Favre-averaged mean free-stream dynamic viscosity",

        # Other quantities
        'Tr': 'Recovery temperature',
        'Bk': 'Kinematic Rotta-Clauser parameter',

        # Non-dimensional numbers
        'Redelta99': 'Visual thickness Reynolds number',
        'Redelta1': 'Displacement thickness Reynolds number',
        'Redelta2': 'Momentum thickness Reynolds number based on wall viscosity',
        'Retheta': 'Momentum thickness Reynolds number based on freestream viscosity',
    }
    
    def __init__(self, units:str, **params):
        # Initialize all the members listed in the BL parameter list
        for key in BL.bl_params.keys():
            setattr(self, key, None)

        # Read in provided parameters and assign those specific to BL type
        poplist = []
        for key, val in params.items():
            if key in BL.bl_params.keys():
                setattr(self, key, val)
                poplist.append(key)
        for popkey in poplist:
            params.pop(popkey)

        super().__init__('bl',units,**params)

    def whatis(p):
        # Return description for parameter p
        if p in BL.bl_params.keys():
            print(BL.bl_params[p])
        else:
            super().whatis(p)

    def hasdata(self, p=None):
        r = []
        if p is None:
            for param in BL.bl_params.keys():
                if getattr(self,param,None) is not None: 
                    r.append(param)
        tmp = super().hasdata(p)        
        if isinstance(tmp, list): r.extend(tmp)
        else: r.append(tmp)
        if len(r) == 1: return r[0] 
        else: return r

    def find_edge(self, edge_type='delta99', interp=True, sigma_smooth=None):
        # TODO: Currently sometimes slightly underestimates values of delta1, delta2 
        # compared to those reported... unsure why or if it matters?
        if self.u.ndim > 1:
            nx, ny = self.u.shape
        else:
            nx, ny = 1, self.u.shape[0]

        U, V, P, RHO = self.u, self.v, self.P, self.rho
        if sigma_smooth is not None:
            U = gaussian_filter(U, sigma=sigma_smooth)
            V = gaussian_filter(V, sigma=sigma_smooth)
            P = gaussian_filter(P, sigma=sigma_smooth)
            RHO = gaussian_filter(RHO, sigma=sigma_smooth)

        if edge_type != 'delta99':
            if not self.hasdata('delta99'):
                self.delta99, _ = self.find_edge('delta99', interp=interp, sigma_smooth=sigma_smooth)
            id99 = np.argmin(np.abs(self.y[np.newaxis,:] - self.delta99[:,np.newaxis]),axis=-1)
            if id99.ndim > 1: id99 = id99.squeeze()

        
        match edge_type:
            case 'delta99':
                # This is best way I could come up with to do this efficiently for 2D arrays
                tmp = np.diff(U<0.99*self.uinf[:,np.newaxis])
                ide = np.argmax(tmp,axis=-1) # First occurence of crossing 0.99uinf
                # Check either side to choose what is closer to 0.99uinf
                ide0 = np.transpose(np.array(list(zip(range(nx),ide))))
                ide1 = np.transpose(np.array(list(zip(range(nx),ide+1))))
                u0 = U[*ide0]; u1 = U[*ide1]

                uu = np.concatenate([u0[:,np.newaxis], u1[:,np.newaxis]], axis=1)
                idei = np.argmin(np.abs(uu - 0.99*self.uinf[:,np.newaxis]), axis=1)
                ide_ret = ide + idei

                if self.y.ndim > 1:
                    # When there are different y values for different x locations (curved walls)
                    id99 = np.transpose(np.array(list(zip(range(nx),ide_ret))))
                    d = self.y[*id99] if not interp else \
                        self.y[*ide0] + (0.99*self.uinf - u0) * (self.y[*ide1] - self.y[*ide0]) / (u1 - u0)
                else:
                    d = self.y[ide_ret] if not interp else \
                        self.y[ide] + (0.99*self.uinf - u0) * (self.y[ide+1] - self.y[ide]) / (u1 - u0)
                return d, ide_ret
            
            case 'deltainf':
                # Not a true BL thickness, but a standardized location to get true freestream values
                # TODO: This is me just kind of choosing something... maybe there is a better way?
                ide = id99 + (ny-id99)//2
                if self.y.ndim > 1:
                    # When there are different y values for different x locations (curved walls)
                    idee = np.transpose(np.array(list(zip(range(nx),ide))))
                    d = self.y[*idee]
                else:
                    d = self.y[ide]
                return d, ide
            
            case 'delta99GFM':
                # Boundary layer thickness based on Griffin, Fu, Moin definition for
                # nonequilibrium flows
                # As best as I can tell this is implemented correctly?
                # Reference values are freestream: appears to be roughly insensitive to particular choice
                idinf = id99 + (ny-id99)//2
                idinf = np.transpose(np.array(list(zip(range(nx),idinf))))
                
                Umref2 = U[*idinf]**2 + V[*idinf]**2
                Pref = P[*idinf]
                rhoref = RHO[*idinf]
                for varref in [Umref2, Pref, rhoref]:
                    if not varref.ndim: varref = varref[np.newaxis]
                
                UI2 = 2.0*self.gamma/(self.gamma-1)*((Pref/rhoref)[:,np.newaxis] - P/RHO)
                UI2 += Umref2[:,np.newaxis] - V**2

                # Find delta based on condition (similar to delta99)
                tmp = np.diff(U**2 < (0.99**2)*UI2)
                ide = np.argmax(tmp,axis=-1) # First occurence of crossing condition
                # Check either side to choose what is closer to condition
                ide0 = np.transpose(np.array(list(zip(range(nx),ide))))
                ide1 = np.transpose(np.array(list(zip(range(nx),ide+1))))
                u0 = U[*ide0]; u1 = U[*ide1]
                U1 = UI2[*ide0]; U2 = UI2[*ide1]

                uu = np.concatenate([u0[:,np.newaxis], u1[:,np.newaxis]], axis=1)
                UU = np.concatenate([U1[:,np.newaxis], U2[:,np.newaxis]], axis=1)
                idei = np.argmin(np.abs(uu**2 - (0.99**2)*UU), axis=1)
                ide_ret = ide + idei

                if self.y.ndim > 1:
                    # When there are different y values for different x locations (curved walls)
                    id99 = np.transpose(np.array(list(zip(range(nx),ide_ret))))
                    d = self.y[*id99] if not interp else \
                        self.y[*ide0] + (0.99*self.uinf - u0) * (self.y[*ide1] - self.y[*ide0]) / (u1 - u0)
                else:
                    d = self.y[ide_ret] if not interp else \
                        self.y[ide] + (0.99*self.uinf - u0) * (self.y[ide+1] - self.y[ide]) / (u1 - u0)
                return d, ide_ret
            
            case 'delta1':
                I = 1-(RHO*U)/np.transpose([(self.rhoinf*self.uinf)])
            
            case 'delta1k':
                I = 1-U/np.transpose([self.uinf])
            
            case 'delta2':
                I = (RHO*U)/np.transpose([(self.rhoinf*self.uinf)])
                I *= (1-U/np.transpose([self.uinf]))
            
            case 'delta2k':
                I = U/np.transpose([self.uinf])
                I *= (1-U/np.transpose([self.uinf]))
            
            case _:
                raise ValueError(f"Unknown edge type: {edge_type}")
        
        I = I.squeeze()
        if nx > 1:
            d = np.zeros([nx])
            for i in range(nx):
                s = np.s_[i,:id99[i]+1] if self.y.ndim > 1 else np.s_[:id99[i]+1]
                d[i] = simpson(I[i,:id99[i]+1], self.y[s])
        else:
            d = np.array([simpson(I[:id99[0]+1], self.y[:id99[0]+1])])
        ide = np.argmin(np.abs(self.y[np.newaxis,:] - d[:,np.newaxis]),axis=-1)
        return d, ide


class Channel(Case):
    channel_params = {
        # Channel geometry
        'h': 'Channel half-height',

        # Non-dimensional numbers
        'Reh': 'Channel half-height Reynolds number',
        'Rebulk': 'Channel bulk Reynolds number',
        'Mbulk': 'Channel bulk Mach number',
    }
    
    def __init__(self, units:str, **params):
        # Initialize all the members listed in the channel parameter list
        for key in Channel.channel_params.keys():
            setattr(self, key, None)

        # Read in provided parameters and assign those specific to Channel type
        poplist = []
        for key, val in params.items():
            if key in Channel.channel_params.keys():
                setattr(self, key, val)
                poplist.append(key)
        for popkey in poplist:
            params.pop(popkey)

        super().__init__('channel',units,**params)

    
    def whatis(p):
        # Return description for parameter p
        if p in Channel.channel_params.keys():
            print(Channel.channel_params[p])
        else:
            super().whatis(p)

    def hasdata(self, p=None):
        r = []
        if p is None:
            for param in Channel.channel_params.keys():
                if getattr(self,param,None) is not None: 
                    r.append(param)
        r.append(super().hasdata(p))
        if len(r) == 1: return r[0] 
        else: return r


class Pipe(Case):
    pipe_params = {

    }
    
    def __init__(self, units:str, **params):
        # Initialize all the members listed in the pipe parameter list
        for key in Pipe.pipe_params.keys():
            setattr(self, key, None)

        # Read in provided parameters and assign those specific to pipe type
        for key, val in params.items():
            if key in Pipe.pipe_params.keys():
                setattr(self, key, val)
                params.pop(key)

        super().__init__('pipe',units,**params)

    
    def whatis(p):
        # Return description for parameter p
        if p in Pipe.pipe_params.keys():
            print(Pipe.pipe_params[p])
        else:
            super().whatis(p)

    def hasdata(self, p=None):
        r = []
        if p is None:
            for param in Pipe.pipe_params.keys():
                if getattr(self,param,None) is not None: 
                    r.append(param)
        r.append(super().hasdata(p))
        if len(r) == 1: return r[0] 
        else: return r


class Duct(Case):
    duct_params = {

    }
    
    def __init__(self, units:str, **params):
        # Initialize all the members listed in the duct parameter list
        for key in Duct.duct_params.keys():
            setattr(self, key, None)

        # Read in provided parameters and assign those specific to duct type
        for key, val in params.items():
            if key in Duct.duct_params.keys():
                setattr(self, key, val)
                params.pop(key)

        super().__init__('duct',units,**params)

    
    def whatis(p):
        # Return description for parameter p
        if p in Duct.duct_params.keys():
            print(Duct.duct_params[p])
        else:
            super().whatis(p)

    def hasdata(self, p=None):
        r = []
        if p is None:
            for param in Duct.duct_params.keys():
                if getattr(self,param,None) is not None: 
                    r.append(param)
        r.append(super().hasdata(p))
        if len(r) == 1: return r[0] 
        else: return r