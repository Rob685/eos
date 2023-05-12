import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

########### calculating entropy #############

def scvh_reader(tab_name):
    tab = []
    head = []
    with open('/Users/Helios/burrows_research/scvh_eos_adam/eos/'+tab_name) as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            if line.startswith("#"):
                line = line.lstrip('#')
                head.append(line)

            else:# j > head: # skipping the header
                tab.append(list(float(line[i:i+8]) for i in range(0, len(line), 8)))

    header = list(filter(None, ' '.join(head).split(' ')))

    tab = np.array(tab)
    tab_ = np.reshape(tab, (300, 100))
    return [float(val) for val in header], tab_

bounds_s, stab = scvh_reader('stabnew_adam.dat')
R1, R2, T1, T2, T11, T12 = bounds_s[2:8] # same bounds

logrho_array = np.linspace(R1, R2, 300)

stab_names = ['s22scz.dat', 'stabnew_adam.dat', 's28scz.dat', 's30.dat']
ptab_names = ['p22scz.dat', 'ptabnew_adam.dat', 'p28scz.dat', 'p30.dat']

stabs = np.array([scvh_reader(name)[-1] for name in stab_names])
ptabs = np.array([scvh_reader(name)[-1] for name in ptab_names])

yarr = np.array([0.22, 0.25, 0.28, 0.30])

def x(R):
    return (R - R1)*(300 - 1)/(R2 - R1)

def deltaT(R):
    return (R - R1)*((T12 - T11) - (T2 - T1))/(R2 - R1) + (T2 - T1)

def T1p(R):
    m1 = (T11 - T1)/(R2 - R1)
    return m1*(R - R1) + T1

def y(R, T):
    return (T - T1p(R))*(100 - 1)/deltaT(R)

def get_tarr(R):
    m1 = (T12 - T1)/(R2 - R1)
    b = T1 - m1*R1
    T1p = R*m1 + b
    #T1p = (R - R1)*m1 + b
    T2p = T1p + deltaT(R)

    return np.linspace(T1p, T2p, 100)


def get_s(R, T, yhe):
     # all of these have the same temperature ranges so I don't need to change the bounds each time
    
    if 0.22 <= yhe < 0.25:
        y1, y2 = 0.22, 0.25
        tab1 = stabs[0]
        tab2 = stabs[1]

    elif 0.25 <= yhe < 0.28:
        y1, y2 = 0.25, 0.28
        tab1 = stabs[1]
        tab2 = stabs[2]
        
    elif 0.28 <= yhe <= 0.30:
        y1, y2 = 0.28, 0.30
        tab1 = stabs[2]
        tab2 = stabs[3]
    
    # x_arr = np.arange(0, 300, 1)
    # y_arr = np.arange(0, 100, 1)
    
    eta1 = (y2 - yhe)/(y2 - y1)
    eta2 = 1 - eta1
    
    smix = eta1*tab1 + eta2*tab2
    
    # x_arr = np.arange(0, 300, 1)
    # y_arr = np.arange(0, 100, 1)

    logt = get_tarr(R)

    try:
        #interp_stab = RGI((yarr, logrho_array, logt), stabs, method='linear', bounds_error=False, fill_value=None)
        interp_stab = RGI((logrho_array, logt), smix, method='linear', bounds_error=False, fill_value=None)
    except:
        print(np.shape(logrho_array))
        raise
    
    #return interp_stab(np.array([yhe, R, T]).T)
    return float(interp_stab((R, T)))

def get_p(R, T, yhe):
     # all of these have the same temperature ranges so I don't need to change the bounds each time
    
    if 0.22 <= yhe < 0.25:
        y1, y2 = 0.22, 0.25
        tab1 = ptabs[0]
        tab2 = ptabs[1]

    elif 0.25 <= yhe < 0.28:
        y1, y2 = 0.25, 0.28
        tab1 = ptabs[1]
        tab2 = ptabs[2]
        
    elif 0.28 <= yhe <= 0.30:
        y1, y2 = 0.28, 0.30
        tab1 = ptabs[2]
        tab2 = ptabs[3]
    
    # x_arr = np.arange(0, 300, 1)
    # y_arr = np.arange(0, 100, 1)
    
    eta1 = (y2 - yhe)/(y2 - y1)
    eta2 = 1 - eta1
    
    pmix = eta1*tab1 + eta2*tab2
    
    # x_arr = np.arange(0, 300, 1)
    # y_arr = np.arange(0, 100, 1)

    logt = get_tarr(R)

    try:
        #interp_stab = RGI((yarr, logrho_array, logt), stabs, method='linear', bounds_error=False, fill_value=None)
        interp_ptab = RGI((logrho_array, logt), pmix, method='linear', bounds_error=False, fill_value=None)
    except:
        print(np.shape(logrho_array))
        raise
    
    #return interp_stab(np.array([yhe, R, T]).T)
    return float(interp_ptab((R, T)))
