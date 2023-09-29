from eos import cms_eos_new as cms_eos
from eos import scvh_eos
import numpy as np
from scipy.interpolate import interp1d


M_jup = 1.89914e30
R_jup = 7.15e9
M_e = 5.9722e27
a = 7.57e-15
c = 3e10
msun = 1.98e33


becker2 = np.genfromtxt("becker_table2.txt")
becker1 = np.genfromtxt("becker_table1.txt")
french1 = np.genfromtxt("French_table1.txt")
french2 = np.genfromtxt("French_table2.txt")

rho_f1 = french1[:,1]
p_f1 = french1[:,3]

p_r_f = interp1d(french1[:,0],french1[:,3]*1e10,kind='cubic',fill_value="extrapolate")
p2 = p_r_f(french2[:,0])
Lambda_c = interp1d(p2,french2[:,3]*1e5,kind='linear',fill_value="extrapolate")


def get_Lambda_cond(peval):
    F2=[]
    for i in range(len(peval)):
        if peval[i]<min(p2):
            F2.append(Lambda_c(p2[-1])*np.log10(peval[i])/np.log10(p2[-1]))
        else:
            F2.append(Lambda_c(peval[i]))
    return np.array(F2)

rad = np.genfromtxt("Henyey240_rad_profiles.txt")
press = np.genfromtxt("Henyey240_press_profiles.txt")
rho = np.genfromtxt("Henyey240_rho_profiles.txt")
temp = np.genfromtxt("Henyey240_temp_profiles.txt")
press[0][0]=1e6

N = len(rad[0])
mesh_surf_amp=1e5
mesh_surf_amp2=1e5
mesh_surf_width=1e-2
mesh_surf_width2=6.5e-3
f0 = np.linspace(0, 1, N)
density_f0 = 1. / np.diff(f0)
density_f0 = np.insert(density_f0, 0, density_f0[0])
density_f0 += mesh_surf_amp * f0 * np.exp((f0 - 1.) / mesh_surf_width) * np.mean(density_f0) # boost mesh density near surface
density_f0 += mesh_surf_amp2 * f0 * np.exp((-f0) / mesh_surf_width2) * np.mean(density_f0)
out = np.cumsum(1. / density_f0)
out -= out[0]
out /= out[-1]
mesh = out    

m = M_jup*mesh[::-1]

dm = np.zeros(N)
dm [1:-1] = (m[0:-2]-m[2:])/2.0
dm[0] = (m[0]-m[1])/2.0
dm[-1] = (-m[-1]+m[-2])/2.0

opa = np.genfromtxt("ross.3.16.default.60")
xgrid = opa[:,0].reshape(100,100)
ygrid = opa[:,1].reshape(100,100)
zgrid = opa[:,2].reshape(100,100)
xgrid_arr = xgrid[:,0]
ygrid_arr = ygrid[0,:]
   
def get_z(x,y,Y):
    
    igl = np.searchsorted(xgrid[:,0],x)-1
    igh = igl+1

    if x>=np.amax(xgrid):
        #print ("outside max temp")
        igl=-2
        igh=-1
    if y<=np.min(ygrid[igl]) or y<=np.min(ygrid[igh]):
        #print ("outside grid in y bottom")
        jl1=0
        jh1=1
        jl2=0
        jh2=1
    elif y>=np.max(ygrid[igl]) or y>=np.max(ygrid[igh]):
        #print ("outisde grid in y top")
        jl1=-2
        jh1=-1
        jl2=-2
        jh2=-1
    else:
        #print ("inside grid")
        jl1=np.where(ygrid[igl]<y)[0][-1]
        jh1=jl1+1
        jl2=np.where(ygrid[igh]<y)[0][-1]
        jh2=jl2+1
                
    
    if x<=np.min(xgrid_arr) and y<=np.max(ygrid_arr) :
        return (zgrid[igh,jh1]+zgrid[igh,jh2])/2.0
    
    if x>=np.max(xgrid_arr) and y<=np.max(ygrid_arr) :
        return (zgrid[igh,jh1]+zgrid[igh,jh2])/2.0
    
    else:
        teff1= np.log10(zgrid[igl,jl1]) + (np.log10(y)-np.log10(ygrid[igl,jl1]))*(np.log10(zgrid[igl,jh1])-np.log10(zgrid[igl,jl1]))/(np.log10(ygrid[igl,jh1])-np.log10(ygrid[igl,jl1]))
        teff2= np.log10(zgrid[igh,jl2]) + (np.log10(y)-np.log10(ygrid[igh,jl2]))*(np.log10(zgrid[igh,jh2])-np.log10(zgrid[igh,jl2]))/(np.log10(ygrid[igh,jh2])-np.log10(ygrid[igh,jl2]))
        teff = teff1+(np.log10(x)-np.log10(xgrid[:,0][igl]))*(teff2-teff1)/(np.log10(xgrid[:,0][igh])-np.log10(xgrid[:,0][igl]))
        return 10**teff

g = 6.6e-8*m[0]/rad[0][0]**2
g_id = np.where(g>ggrid[:,0])[0][-1]
y = interp1d(sgrid[g_id],tingrid[g_id],kind='cubic',fill_value="extrapolate")
yinv = interp1d(tingrid[g_id],sgrid[g_id],kind='cubic',fill_value="extrapolate")

def get_tint(g,t10,S_,bc_atm):
    if bc_atm=="fortney2011a":
        Tint = atm_boundary.Teff(g,t10,bc_atm)
    if bc_atm=="fortney2011b":
        Tint = atm_boundary.Teff(g,t10,bc_atm)
    if bc_atm=="gT1fit":
        if Tint_old>200:
            Tint = (g**0.167*t10/15.86)**(1.0/0.95)
            Tint_old=Tint
        else:
            Tint = (g**0.167*t10/3.36)**(1.0/1.243)
                   
    if bc_atm=="Burrows":
        Tint = atm_boundary.Tint(g,t10,bc_atm)   
    if bc_atm=="Yixian":
        Tint=atm_boundary.Tint(g,S_,bc_atm) 
    return Tint

temp0 = temp[0]
temp_old = temp0
dt_ = 1e2
t = dt_
i=0
Y = 0.25
#S0 =  10**scvh_eos.get_s(np.log10(rho[0]),np.log10(temp_old),np.zeros(N)+Y) 
Hp = press[0]/rho[0]/g
norm = 83144626.2102654

### thermal evolution function, Schwarzschild condition for now

def thermal(T,r,P,rho_,m,dm,T_int,dt,t,Y,S,losses=True,convection=False,radiation=True,weight=False,implicit_rad=False):   

    #Cv = scvh_eos.get_c_v(S,np.log10(press[0]),np.zeros(N)+Y)
    #Cp = scvh_eos.get_c_p(S,np.log10(press[0]),np.zeros(N)+Y)
    
    Cv = cms_eos.get_c_v(S,np.log10(press[0]),np.zeros(N)+Y)
    Cp = cms_eos.get_c_p(S,np.log10(press[0]),np.zeros(N)+Y)
    
    
    #nabla_ad = (scvh_eos.get_grad_ad(S,np.log10(press[0]),np.zeros(N)+Y))
    #gamma1 = scvh_eos.get_gamma_1(S,np.log10(press[0]),np.zeros(N)+Y)
    #gamma = gamma1*nabla_ad

    alpha = 1.0   
    sigma_cd = 4*np.pi*r**2*(Lambda_c(P))
    ross_k = get_z_vect(T,rho_,Y)
        
    ross_k_f = interp1d(T,ross_k,fill_value='extrapolate')
    Lambda_rad = 4*a*c*T**3/ross_k/3.0/rho_
    Lambda_rad_func = interp1d(T,Lambda_rad,fill_value='extrapolate')
    #Lambda = (Lambda_c(P) + Lambda_rad)
    Lambda = Lambda_rad + get_Lambda_cond(P)
    Lambda = Lambda*T/Cv

    #alp = (Cp-Cv)/Cv/gamma/T
    S_cgs = S*norm
        
    if convection:
        if weight:
            #phi = get_weight()*4*np.pi*r**2*(rho_*P*T*nabla_ad/g)**1.5
            phi = get_weight()*4*np.pi*r**2*rho_*T*(g*Hp**4/Cp/32)**(0.5)
        else:
            #phi = 4*np.pi*r**2*(rho_*P*T*nabla_ad/g)**1.5
            phi = 4*np.pi*r**2*rho_*T*(g*Hp**4/Cp/32)**(0.5)
    else:
        phi = np.zeros(N)
    
    
    A = np.zeros((N,N))
    B = np.zeros(N)
    
    A[0][0] = T[0]*dm[0]/dt
    dSdr_p = (S_cgs[1]-S_cgs[0])/(r[1]-r[0])
    if dSdr_p<0:
        B[0] = (phi[0]+phi[1])/2.0*(-dSdr_p)**(1.5)
        A[0][1] = (phi[0]+phi[1])/2.0*1.5*(-dSdr_p)**(0.5)/(r[1]-r[0])
        A[0][0] += -(phi[0]+phi[1])/2.0*1.5*(-dSdr_p)**(0.5)/(r[1]-r[0])
    
    
    if radiation:
        Frad_p = 4*np.pi*(r[0]+r[1])**2/4.0*(Lambda[0]+Lambda[1])/2.0/(r[1]-r[0])
        A[0][1]+=Frad_p 
        A[0][0]+=- Frad_p
        B[0]+=-Frad_p*(S_cgs[1]-S_cgs[0])
        
        # if implicit_rad:
        #     A[0][1]+=0.5*np.pi*(r[0]+r[1])**2*dlambda_dT2(T[1],ross_k_f,rho_[1])*(T[1]-T[0])/(r[1]-r[0])
        #     A[0][0]+=0.5*np.pi*(r[0]+r[1])**2*dlambda_dT2(T[0],ross_k_f,rho_[0])*(T[1]-T[0])/(r[1]-r[0])
        
    
    if losses:
        dTdS = 0.1/(yinv(T_int+0.1)-yinv(T_int))
        A[0][0]+=np.pi*(r[0]+r[1])**2*const.sigma_sb*4*T_int**3*dTdS/norm
        B[0] = B[0]- np.pi*(r[0]+r[1])**2*const.sigma_sb*T_int**4
        
    
    for k in range(1,N-1):
        
        
        dSdr_p = (S_cgs[k+1]-S_cgs[k])/(r[k+1]-r[k])
        dSdr_m = (S_cgs[k]-S_cgs[k-1])/(r[k]-r[k-1])
        A[k][k] = T[k]*dm[k]/dt
        
        if dSdr_p<0:
            B[k] = (phi[k]+phi[k+1])/2.0*(-dSdr_p)**(1.5)
            A[k][k+1] = (phi[k]+phi[k+1])/2.0*1.5*(-dSdr_p)**(0.5)/(r[k+1]-r[k])
            A[k][k] += -(phi[k]+phi[k+1])/2.0*1.5*(-dSdr_p)**(0.5)/(r[k+1]-r[k])
            
        if dSdr_m<0:
            B[k]+= -(phi[k]+phi[k-1])/2.0*(-dSdr_m)**(1.5)
            A[k][k-1] = (phi[k]+phi[k-1])/2.0*1.5*(-dSdr_m)**(0.5)/(r[k]-r[k-1])
            A[k][k]+= -(phi[k]+phi[k-1])/2.0*1.5*(-dSdr_m)**(0.5)/(r[k]-r[k-1])
        
        if radiation:
            Frad_m = 4*np.pi*(r[k]+r[k-1])**2/4.0*(Lambda[k]+Lambda[k-1])/2.0/(r[k]-r[k-1])
            Frad_p = 4*np.pi*(r[k]+r[k+1])**2/4.0*(Lambda[k]+Lambda[k+1])/2.0/(r[k+1]-r[k])
            A[k][k-1] += Frad_m 
            A[k][k+1] += Frad_p
            A[k][k] += - Frad_p - Frad_m 
            B[k] += -Frad_p*(S_cgs[k+1]-S_cgs[k])+Frad_m*(S_cgs[k]-S_cgs[k-1])
            
            # if implicit_rad:
            #     A[k][k-1]+= -0.5*np.pi*(r[k]+r[k-1])**2*dlambda_dT2(T[k-1],ross_k_f,rho_[k-1])*(T[k]-T[k-1])/(r[k]-r[k-1])
            #     A[k][k+1]+= 0.5*np.pi*(r[k]+r[k+1])**2*dlambda_dT2(T[k+1],ross_k_f,rho_[k+1])*(T[k+1]-T[k])/(r[k+1]-r[k])
            #     A[k][k]+= 0.5*np.pi*(r[k]+r[k+1])**2*dlambda_dT2(T[k],ross_k_f,rho_[k])*(T[k+1]-T[k])/(r[k+1]-r[k])\
            #      - 0.5*np.pi*(r[k]+r[k-1])**2*dlambda_dT2(T[k],ross_k_f,rho_[k])*(T[k]-T[k-1])/(r[k]-r[k-1])
                

    A[N-1][N-1] = T[N-1]*dm[N-1]/dt
    
    dSdr_m = (S_cgs[N-1]-S_cgs[N-2])/(r[N-1]-r[N-2])
    if dSdr_m<0:
        A[N-1][N-1] += -(phi[N-1]+phi[N-2])/2.0*1.5*(-dSdr_m)**(0.5)/(r[N-1]-r[N-2])
        A[N-1][N-2] += (phi[N-1]+phi[N-2])/2.0*1.5*(-dSdr_m)**(0.5)/(r[N-1]-r[N-2])
        B[N-1] += -(phi[N-1]+phi[N-2])/2.0*(-dSdr_m)**(1.5)

    
    if radiation:
        Frad_m = 4*np.pi*(r[N-1]+r[N-2])**2/4.0*(Lambda[N-1]+Lambda[N-2])/2.0/(r[N-1]-r[N-2])
        A[N-1][N-2] += Frad_m
        A[N-1][N-1] += -Frad_m 
        B[N-1] += Frad_m*(S_cgs[N-1]-S_cgs[N-2])
        
        # if implicit_rad:
        #     A[N-1][N-2]+=- 0.5*np.pi*(r[N-1]+r[N-2])**2*dlambda_dT2(T[N-2],ross_k_f,rho_[N-2])*(T[N-1]-T[N-2])/(r[N-1]-r[N-2])
        #     A[N-1][N-1]+=- 0.5*np.pi*(r[N-1]+r[N-2])**2*dlambda_dT2(T[N-1],ross_k_f,rho_[N-1])*(T[N-1]-T[N-2])/(r[N-1]-r[N-2])
       
        
    Sn = S_cgs.copy() 
    dS = spsolve(A,B)
        
    S_cgs = Sn + dS
    Tnew = T*np.exp(dS/Cv)
    return S_cgs/norm, Tnew