import numpy as np
from scipy.optimize import brentq

G_N = 6.67408e-11 * 6.7702543e-20  # pc^3 Msun^-1 s^-2


class elliptic_orbit:

    def __init__(self, a, e, galaxy):

        self.a = a
        self.e = e
        self.Galaxy = galaxy

        self.M_enc = galaxy.M_enc(a)
        self.T_orb = 2 * np.pi * np.sqrt(a**3 / (G_N * self.M_enc))


    # ===============================
    # Kepler solver
    # ===============================

    def calc_M_anom(self, t):
        return 2 * np.pi * ((t % self.T_orb) / self.T_orb)


    def calc_E(self, M):
        """
        Solve Kepler equation:
        M = E - e sin E
        Vectorized via Newton-Raphson
        """

        M = np.asarray(M)

        # Good initial guess
        E = M.copy()

        for _ in range(5):  # 5 iterations sufficient for e < 0.99
            f  = E - self.e * np.sin(E) - M
            fp = 1 - self.e * np.cos(E)
            E -= f / fp

        return E


    # ===============================
    # Orbital quantities
    # ===============================

    def calc_r(self, t):
        M = self.calc_M_anom(t)
        E = self.calc_E(M)
        return self.a * (1 - self.e * np.cos(E))


    def calc_theta(self, t):
        """
        Stable true anomaly formula:
        tan(theta/2) = sqrt((1+e)/(1-e)) * tan(E/2)
        """

        M = self.calc_M_anom(t)
        E = self.calc_E(M)

        sinE = np.sin(E)
        cosE = np.cos(E)

        theta = np.arctan2(
            np.sqrt(1 - self.e**2) * sinE,
            cosE - self.e
        )

        return theta


    def vis_viva_r(self, r):
        return np.sqrt(G_N * self.M_enc * (2/r - 1/self.a))


    def vis_viva_t(self, t):
        r = self.calc_r(t)
        return self.vis_viva_r(r)
