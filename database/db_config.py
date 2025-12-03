import dill
import numpy as np
from warnings import warn

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

    # List of all parameters with description
    gen_params = {
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

        # Scaling
        "deltaplus": "Friction length scale",
        "deltastar": "Semi-local length scale",
        "yplus": "Wall-normal coordinate in friction units",
        "ystar": "Wall-normal coordinate in semi-local units",
        "uplus": "Streamwise velocity in frction units",

        # Non-dimensional numbers
        "Pr": "Molecular Prandtl number",
        "Prt": "Reynolds Turbulent Prandtl number",
        "Prt_F": "Favre Turbulent Prandtl number",
        "Retau": "Friction Reynolds number",
        "Retaustar": "Semi-local friction Reynolds number",
    }
    __frozen = 0

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
        if self.__frozen and not hasattr(self, key):
            raise AttributeError("%r is frozen: cannot add additional members after " \
            "initialization" % self)
        super().__setattr__(key, value)

    def _freeze(self):
        # Freeze object so can't add new attributes
        self.__frozen = 1

    def _unfreeze(self):
        # Unfreeze object so can add new attributes: should never need to use!
        self.__frozen = 0

    def whatis(p):
        # Return description for parameter p
        if p in Case.gen_params.keys():
            print(Case.gen_params[p])
        else:
            warn("%r is not a valid case parameter" % p)

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
        prep = True if self.y[0]>0 else False
        if label:
            self._unfreeze()

            # Check if label corresponds to one of known transformations
            if label.upper() == "VD":
                # Van Driest
                f = np.ones(self.u.shape)
                g = np.sqrt(self.rho/self.rhow)
            elif label.upper() == "TL":
                # Trettel & Larsson
                f = np.gradient(self.y*np.sqrt(self.rho/self.rhow)/(self.mu/self.muw),self.y,axis=-1)
                g = self.mu/self.muw * f
            elif label.upper() == "V":
                # Volpiani et al.
                f = np.sqrt(self.rho/self.rhow)/(self.mu/self.muw)**(3.0/2.0)
                g = np.sqrt(self.rho/self.rhow)/np.sqrt(self.mu/self.muw)
            elif label.upper() == "GFM":
                # Griffin, Fu, Moin
                pass
            elif label.upper() == "H":
                # Hasan et al.
                pass

        if f is None or g is None:
            raise ValueError("Transformation is not known and no kernels were provided")

        if prep:
            f = np.insert(f,0,fw,axis=1 if ndim>1 else 0)
            g = np.insert(g,0,gw,axis=1 if ndim>1 else 0)    

        # Perform the numerical integration using given kernels
        yscaled, uscaled = np.zeros(self.y.shape), np.zeros(self.u.shape)
        sj = np.s_[:,0] if ndim>1 else np.s_[0]
        sjp1 = np.s_[:,1] if ndim>1 else np.s_[1]
        uscaled[sj] = (g[sj]+g[sjp1])/2.0 * self.u[sj]
        yscaled[sj] = (f[sj]+f[sjp1])/2.0 * self.y[sj]

        if prep:
            # Cut out prepended wall entry to simplify slicing
            g = g[1:] if ndim==1 else g[:,1:]
            f = f[1:] if ndim==1 else f[:,1:]

        for j in range(1, len(self.y)):
            uscaled[sjp1] = uscaled[sj] + (g[sj]+g[sjp1])/2.0*(self.u[sjp1]-self.u[sj])
            yscaled[sjp1] = yscaled[sj] + (f[sj]+f[sjp1])/2.0*(self.y[sjp1]-self.y[sj])
            sj = sjp1
            sjp1 = np.s_[:,j+1] if ndim>1 else np.s_[j+1]

        if label is not None:
            setattr(self, f'y{label}', yscaled)
            setattr(self, f'u{label}', uscaled)
            setattr(self, f'yplus{label}', yscaled/self.deltaplus)
            setattr(self, f'uplus{label}', uscaled/self.utau)
            self._freeze()
        return yscaled, uscaled


class BL(Case):
    bl_params = {
        # BL Geometry
        'delta99': 'Visual thickness',
        'delta1': 'Displacement thickness',
        'delta1k': 'Kinematic displacement thickness',
        'delta2': 'Momentum thickness (often written as theta)',
        'H': 'Shape factor',

        # Edge/free-stream quantities
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
        r.append(super().hasdata(p))
        if len(r) == 1: return r[0] 
        else: return r


class Channel(Case):
    channel_params = {
        # Channel geometry
        'h': 'Channel half-height',

        # Non-dimensional numbers
        'Reh': 'Channel half-height Reynolds number',
    }
    
    def __init__(self, units:str, **params):
        # Initialize all the members listed in the channel parameter list
        for key in Channel.channel_params.keys():
            setattr(self, key, None)

        # Read in provided parameters and assign those specific to channel type
        for key, val in params.items():
            if key in Channel.channel_params.keys():
                setattr(self, key, val)
                params.pop(key)

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