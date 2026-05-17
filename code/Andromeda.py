import numpy as np

import dirs

G    = 4.302275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun

#Average stellar mass
#This number is not really based on anything concrete...
M_star_avg = 1.0 #M_sun

# Bulge distribution from Tamm et al. 1208.5712
# R and Z are spherical coordinate systems in pc
def rho_star_bulge(R, Z):
    rc = 2.025e3         #pc
    dN = 11.67
    q  = 0.73
    rho_star_core = 0.22 #Msun/pc^3
    rp  = np.sqrt(R**2 + (Z/q)**2)
    # Einasto profile, Eq.2 of 1208.5712
    return rho_star_core*np.exp(-dN*((rp/rc)**0.25 - 1.))

def rho_star_disc(R, Z):
    rc = 11.35e3         #pc
    dN = 2.67
    q  = 0.1
    rho_star_disc = 0.0172 #Msun/pc^3
    rp  = np.sqrt(R**2 + (Z/q)**2)
    return rho_star_disc*np.exp(-dN*(rp/rc - 1.))

def rho_star(R, Z):
    return rho_star_bulge(R, Z) + rho_star_disc(R, Z)



#----------- Enclosed mass and velocity dispersion
def rhoNFW(R):
    rho0 = 5.0e6 * 1e-9  # Msun*pc^-3, see astro-ph/0110390
    rs = 25.0e3  # pc, see astro-ph/0110390 table 3 using virial radius/concentration
    aa = R / rs
    return rho0 / aa / (1 + aa) ** 2

def M_enc(r):
    rho0 = 5.0e6 * 1e-9  # Msun pc^-3, see astro-ph/0110390
    rs = 25.0e3  # pc
    M = 4 * np.pi * rho0 * rs ** 3 * (np.log((rs + r) / rs) - r / (rs + r))
    M_BH = 3e7
    return M + M_BH

# Velocity dispersion at a given radius rho
def sigma(r):
    r_clip = np.clip(r, 1e-20, 1e20)
    return np.sqrt(G * (M_enc(r_clip)) / r_clip)  # km/s

# Local circular speed
def Vcirc(Mstar, r):
    return np.sqrt(G_pc * (Mstar + Menc(r)) / r)  # pc/s



