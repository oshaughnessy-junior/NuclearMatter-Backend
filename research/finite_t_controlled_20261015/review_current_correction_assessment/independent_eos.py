"""Review-only independent Bessel/p-momentum EOS; pure functions only."""
import math
import numpy as np
from types import SimpleNamespace
from scipy.special import kve,expit
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
HC=197.3269804
C=SimpleNamespace(MN=939.56542052,MP=938.27208816,ME=.51099895069)
# Extracted pure functions from prior independent EOS source SHA d28c2bc0bc92d1826155ff46e8bce28ef381c82934aff88d9932ab45c0af6126
def bessel_density(T, eta, m):
    k = np.arange(1, 33)
    return m * m * T / (math.pi ** 2 * HC ** 3) * np.sum((-1.0) ** (k + 1) * np.exp(k * eta) * kve(2, k * m / T) / k)

def baryon(T, n, m):
    eta = brentq(lambda z: bessel_density(T, z, m) / n - 1, -100.0, -0.001, xtol=2e-13, rtol=1e-14)
    mu = m + T * eta
    scale = math.sqrt(m * T)

    def fun(q):
        p = scale * q
        E = math.hypot(p, m)
        kin = p * p / (E + m)
        f = expit(eta - kin / T)
        w = f * (1 - f)
        return q * q * np.array([f, E * f, w, E * w])
    vals = scale ** 3 / (math.pi ** 2 * HC ** 3) * quad_vec(fun, 0.0, 30.0, epsabs=0.0, epsrel=3e-12)[0]
    return dict(mu=mu, eta=eta, density=vals[0], energy=vals[1], epsilon=vals[3] / vals[2], chi=vals[2] / T)

def electron(T, mu, order=80):
    x, w = leggauss(order)
    data = np.zeros(8)
    for lo, hi in zip([0.0, 1.0, 4.0, 10.0, 30.0], [1.0, 4.0, 10.0, 30.0, 90.0]):
        p = T * (lo + (x + 1) * (hi - lo) / 2)
        E = np.hypot(p, C.ME)
        a = expit((mu - E) / T)
        b = expit((-mu - E) / T)
        wa = a * (1 - a)
        wb = b * (1 - b)
        weight = w * (hi - lo) * T / 2 * p * p / (math.pi ** 2 * HC ** 3)
        data += np.array([weight @ (a - b), weight @ (E * (a + b)), weight @ (wa + wb), weight @ (E * (wa - wb)), weight @ a, weight @ b, weight @ wa, weight @ wb])
    return dict(density=data[0], energy=data[1], chi=data[2] / T, epsilon=data[3] / data[2], nminus=data[4], nplus=data[5], minus_fraction=data[6] / data[2], plus_fraction=data[7] / data[2])

def eos(T, n, y):
    neutron = baryon(T, n * (1 - y), C.MN)
    proton = baryon(T, n * y, C.MP)
    mu = brentq(lambda z: electron(T, z)['density'] / (n * y) - 1, 0.0, 100.0, xtol=2e-13, rtol=1e-14)
    lepton = electron(T, mu)
    return (dict(T=T, mu_n=neutron['mu'], mu_p=proton['mu'], mu_e=mu), dict(neutron=neutron, proton=proton, electron=lepton, energy=neutron['energy'] + proton['energy'] + lepton['energy'], uY=proton['epsilon'] - neutron['epsilon'] + lepton['epsilon']))
