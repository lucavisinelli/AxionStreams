import numpy as np

kmtopc = 1.0/(3.086*10**13)
MNS  = 1.4 # Msun
RNS  = 10*kmtopc # pc
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.special import erfi
from scipy.special import gamma as gamma_fun

import dirs

G    = 4.302275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun


#Average stellar mass
#This number is not really based on anything concrete...
M_star_avg = 1.0 #M_sun

#Bulge distribution from McMillan, P.J. 2011, MNRAS, 414, 2446 1102.4340
# R and Z are spherical coordinate systems in pc
def rho_star_bulge(R, Z):
    r0 = 75.             #pc ### This value has been changed from 750pc to 75pc
    rc = 2.1e3          #pc
    rho_star_core = 99.3 #Msun/pc^3 ### This value has been changed from 200Msun/pc^3 to 100Msun/pc^3
    rp2 = R**2 + 4*Z**2
    rp  = np.sqrt(rp2)

    return rho_star_core*np.exp(-rp2/rc**2)/(1+rp/r0)**(1.8)

def rho_star_disc(R, Z):
    hZt = 0.3e3 #pc
    hRt = 2.9e3 #pc
    
    hZT = 0.9e3 #pc
    hRT = 3.31e3 #pc

    rho_0t = 1.361    #Msun/pc^3 # Changed from  1.57
    rho_0T = 0.116  #Msun/pc^3 # Changed from 0.0546

    Za  = np.abs(Z)
    return rho_0t*np.exp(-R/hRt - Za/hZt) + rho_0T*np.exp(-R/hRT - Za/hZT)

def rho_star_halo(R, Z):
    nH = 2.77
    qH = 0.64
    rho_star_halo = 4.45e-4 #Msun/pc^3 # Changed from 5.25e-5
    
    Rsun = 8.3e3 #pc
    
    return rho_star_halo*(Rsun/np.sqrt(R**2 + (Z/qH)**2))**nH

def rho_star(R, Z):
    return rho_star_bulge(R, Z) + rho_star_disc(R, Z)# + rho_star_halo(R, Z)


#--- Enclose mass and velocity dispersion
## NFW profile for AMC distribution
def rhoNFW(R):
    rho0 =  1.4e7*1e-9 # Msun*pc^-3, see Table 1 in 1304.5127
    rs = 16.1e3      # pc
    aa = R/rs
    return rho0/aa/(1+aa)**2

def M_enc(r):
    rho0 =  1.4e7*1e-9 # Msun pc^-3, see Table 1 in 1304.5127
    rs   = 16.1e3      # pc
    
    #MW mass enclosed within radius a
    Menc = 4*np.pi*rho0*rs**3*(np.log(1+r/rs) - r/(rs+r)) 
    M_BH = 4e6
    return Menc + M_BH

#Velocity dispersion at a given radius r
def sigma(r):
    r_clip = np.clip(r, 1e-20, 1e20)
    return np.sqrt(G*(M_enc(r_clip))/r_clip) # km/s
    
#Local circular speed
def Vcirc(Mstar, rho):
    return np.sqrt(G_pc*(Mstar+M_enc(r))/r) # pc/s
