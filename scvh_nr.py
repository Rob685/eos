import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
import pdb

erg_to_kbbar = 1.202723550011625e-08

def scvh_reader(tab_name):
    tab = []
    head = []
    with open('state/scvh/eos/'+tab_name) as file:
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

# bounds_s, SL = scvh_reader('stabnew_copy.dat')
# _, PL = scvh_reader('ptabnew_copy.dat')
bounds_s, SL = scvh_reader('s22scz.dat')
_, PL = scvh_reader('p22scz.dat')
# bounds_s, SL = scvh_reader('s28scz.dat')
# _, PL = scvh_reader('p28scz.dat')
# bounds_s, SL = scvh_reader('s30.dat') # 0.30 gives bad results
# _, PL = scvh_reader('p30.dat')
# bounds_s, SL = scvh_reader('s32.dat')
# _, PL = scvh_reader('p32.dat')

yhe, INDEX, R1, R2, T1, T2, T12, T22 = bounds_s
logrho_array = np.linspace(R1, R2, SL.shape[0])

# def deltaT(R):
#     a = (T22 - T12 - (T2 - T1))/(R2 - R1)
#     return (a*(R - R1)) + T2 - T1

# def get_tarr(R):
#     m1 = (T12 - T1)/(R2 - R1)
#     b = T1 - m1*R1
#     T1p = R*m1 + b
#     #T1p = (R - R1)*m1 + b
#     T2p = T1p + deltaT(R)

#     return np.linspace(T1p, T2p, 100)

def deltaT(R):
    eta = (R - R1)/(R2 - R1)
    return eta*(T2 - T1) + (1-eta)*(T22 - T12)

def get_tarr(R):
    m1 = (T12 - T1)/(R2 - R1)
    b = T1 - m1*R1
    T1p = R*m1 + b
    #T1p = (R - R1)*m1 + b
    T2p = T1p + deltaT(R)

    return np.linspace(T1p, T2p, 100)

def get_SP(R, T, method='RGI'):
    logt_array = get_tarr(R)

    if method == 'RBS':
        interp_s = RBS(logrho_array, logt_array, SL)
        interp_p = RBS(logrho_array, logt_array, PL)

        return interp_s.ev(R, T), interp_p.ev(R, T)

    elif method == 'RGI':
        interp_s = RGI((logrho_array, logt_array), SL, method='linear', bounds_error=False, fill_value=None)
        interp_p = RGI((logrho_array, logt_array), PL, method='linear', bounds_error=False, fill_value=None)  

        return interp_s((R, T)), interp_p((R, T))

def get_closest(S_val, P_val, closeval='S'):

    near_id_s = np.unravel_index((np.abs(SL - S_val)).argmin(), SL.shape)
    near_id_p = np.unravel_index((np.abs(PL - P_val)).argmin(), PL.shape)

    logrho_array = np.linspace(R1, R2, SL.shape[0])
    
    if closeval == 'S':
        near_id = near_id_s
    elif closeval == 'P':
        near_id = near_id_p

    rho_near = logrho_array[near_id[0]]
    
    logT_array = get_tarr(rho_near)

    T_near = logT_array[near_id[1]]

    return rho_near, T_near

def get_sp_sixpt(rho_val, T_val):
    # edges
    alpha = T1 + (rho_val - R1)/(R2 - R1) * (T12 - T1)
    beta = T2-T1+((T22-T12)-(T2-T1))*(rho_val-R1)/(R2-R1)
    
    
    QL = (T_val - alpha)/beta
    delta = (rho_val - R1)/(R2 - R1) * 300.0
    
    #print(rho_val, delta)
    JR = int(delta)
    JQ = int(100*QL)
    
    A, B, C, D, E, F = 1, 1, 1, 1, 1, 1
    #try:
        # handling boundaries
    if ((JR > INDEX-2) and (JQ > 98)): 
        JR, JQ = int(INDEX-2), 98
        P = delta - JR
        Q = 100*QL - JQ

    elif ((JR > INDEX-2) and (JQ < 0)): 
        JR, JQ = int(INDEX-2), 1
        P = delta - JR
        Q = 100*QL - JQ

    elif ((JR < 0) and (JQ > 98)): 
        JR, JQ = 1, 98
        P = delta - JR
        Q = 100*QL - JQ

    elif ((JR < 0) and (JQ < 0)): 
        JR, JQ = 1, 1
        P = delta - JR
        Q = 100*QL - JQ

    elif JR < 0: 
        JR = 1
        P = delta - JR
        Q = 100*QL - JQ

    elif JR > INDEX-2: 
        JR = int(INDEX-2)
        P = delta - JR
        Q = 100*QL - JQ

    elif JQ < 0: 
        JQ = 1
        P = delta - JR
        Q = 100*QL - JQ

    elif JQ > 98: 
        JQ = 98
        P = delta - JR
        Q = 100*QL - JQ
    else:
        P = delta - JR
        Q = 100*QL - JQ

    FS = A*0.5*Q*(Q-1.)*SL[JR,JQ-1] + B*0.5*P*(P-1.)*SL[JR-1,JQ]+ C*(1.+P*Q-P*P-Q*Q)*SL[JR,JQ]\
        + D*0.5*P*(P-2.*Q+1.)*SL[JR+1,JQ]+ E*0.5*Q*(Q-2.*P+1.0)*SL[JR,JQ+1]+ F*P*Q*SL[JR+1,JQ+1]

    FP = 0.5*Q*(Q-1.)*PL[JR,JQ-1]+ 0.5*P*(P-1.)*PL[JR-1,JQ]+ (1.+P*Q-P*P-Q*Q)*PL[JR,JQ]\
        + 0.5*P*(P-2.*Q+1.)*PL[JR+1,JQ]+ 0.5*Q*(Q-2.*P+1.0)*PL[JR,JQ+1]+ P*Q*PL[JR+1,JQ+1]

    return FS, FP

# def x(R):
#     return (R - R1)*(300 - 1)/(R2 - R1)

# def logT1(R):
#     return ((R - R1)/(R2 - R1)) + 2

# def y(R, T):
#     return (T - logT1(R))*(100 - 1)/deltaT(R)

# def get_SP(R, T):
#     logt_array = get_tarr(R)
#     xarr = x(logrho_array)
#     yarr = y(R, logt_array)
#     # interp_s = RBS(logrho_array, logt_array, SL)
#     # interp_p = RBS(logrho_array, logt_array, PL)
#     interp_s = RBS(xarr, yarr, SL)
#     interp_p = RBS(xarr, yarr, PL)

#     return float(interp_s.ev(x(R), y(R,T))), float(interp_p.ev(x(R), y(R,T)))


def newton_raphson(S_val, P_val, tol = 1e-6, interp_type='spt'):
    #if interp_type == 'spt':
    rho_old, T_old = get_closest(S_val, P_val, closeval='S') # S and P in cgs

    # elif interp_type == 'rbs':
    #     rho_old = logrho_array[np.argmin(np.abs(SL[:,0] - S_val))]
    #     T_old = 2.1
    error = 1.0
    i = 0
    j = 0
    k = 0
    XF = 1.0

    if interp_type=='spt':
        interp = get_sp_sixpt
    elif interp_type=='rbs':
        interp = get_SP
    
    while error > tol:
        #eps = 0.1
        eps = 0.1 # for rbs testing
        # if i > 5:
        #     interp = get_SP
        # else: 
        #     interp == get_sp_sixpt

        S0, P0 = interp(rho_old, T_old) # returns S, P in cgs
        
        S1, P1 = interp(rho_old*(1 + eps), T_old)
        S2, P2 = interp(rho_old, T_old*(1 + eps))
        #pdb.set_trace()

        DPDR = (P1 - P0)/(eps*(rho_old))
        DPDT = (P2 - P0)/(eps*(T_old))
        DSDT = (S2 - S0)/(eps*(T_old))
        DSDR = (S1 - S0)/(eps*(rho_old))
 
        
        DEN = DPDR*DSDT-DPDT*DSDR

        
        DRXX = (((S0 - S_val)*DPDT + (P_val - P0)*DSDT)/(DEN*(rho_old)))*XF
        DTXX = (((S_val - S0)*DPDR - (P_val - P0)*DSDR)/(DEN*(T_old)))*XF
        if interp_type == 'spt':
            if i > 10:
                XF = 0.5
                i = 0
                
                
            if j > 20:
                XF = 0.3
                j = 0 
                
            if k > 30:
                raise Exception('Timeout for S = {}, logP = {}'.format(10**S_val, P_val))

        elif interp_type == 'rbs':
            if i > 10:
                raise Exception('Timeout for S = {}, logP = {}'.format(10**S_val, P_val))
                #pdb.set_trace()
        delta_rho = DRXX*rho_old
        delta_T = DTXX*T_old
        
        rho_old += delta_rho
        T_old += delta_T
        error = np.min([abs(DRXX), abs(DTXX)])
        i += 1
        j += 1
        k += 1
        print('Indices, error:',i, j, k, error)

    # cp = (10**T_old)*DEN/DPDR
    # cv = (10**T_old)*DSDT

    cp = ((10**S_val)/erg_to_kbbar) * DEN/DPDR
    cv = ((10**S_val)/erg_to_kbbar) * DSDT
    chirho = DPDR # at const T
    chit = DPDT # at const rho
    #grada = (cp - cv)/cp/DPDT
    #gamma1 = (cp/cv) * DPDR
        
    #print('Converged!')
    print('\n')
    #return 10**rho_old, 10**T_old
    return rho_old, T_old, cp, cv, chirho, chit
        

    
    
# res = newton_raphson(np.log10(6), 6.5, interp_type='spt')
# print('Six Pt rho, temp = {} {}'.format(res[0], res[1]))

# res = newton_raphson(np.log10(6), 6.5, interp_type='rbs')
# print('RBS rho, temp = {} {}'.format(res[0], res[1]))

#S = 8.9998, logP = 2.938759321580604
# print(np.log10(5.0), 6)
# get_closest(np.log10(5.0), 6)
# get_sp_sixpt(3.3076923076923084, 5.598560859430425)