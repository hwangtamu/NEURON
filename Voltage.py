__author__ = 'HanWang'

import math
import numpy as np
from matplotlib import pyplot as plt


class Unit:
    diameter = 0
    length = 0
    input_voltage = np.array([])
    in_resistance = 0
    m_resistance = 0
    m_capacitance = 0
    c_resistance = 0
    path_length = 0  # effective distance to hillock
    space_constant = 0
    time_constant = 0

    def __init__(self, d, l, r_m, r_c, c_m, dist=0):
        self.diameter = d
        self.length = l
        self.in_resistance = math.sqrt(r_c*r_m)/2
        self.m_resistance = r_m/(math.pi*d*l)
        self.m_capacitance = math.pi*d*l*c_m
        self.c_resistance = 4*l*r_c/(math.pi*d*d)
        self.path_length = dist
        self.space_constant = math.sqrt(d*r_m*r_c/4)
        self.time_constant = r_m*c_m

    def add_current(self, current):
        self.input_voltage = current*self.in_resistance

# =========================== SET DATA ===========================
dendrite = []
n = 100
a = 5
rc = 200
rm = 20000
cm = 1  # time constant = 20000*1 = 20000 microseconds = 20 milliseconds
length = 0.1 # length of each compartment
time = np.arange(0, 10, 0.01)
hillock = Unit(0.0001, 0.1, 20000, 200, 1)

# sample data:
# diameters of compartments default = 0.0001 (cm)
diameters = [0.00008, 0.00009, 0.0001, 0.00011]
# synapse inputs (0 = no input, other values = magnitude)
# the injection vectors can be assigned to particular dendrite compartments
injection = np.matrix([[2, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 2, 0, 0, 1, 0]])
# ========================= SET DENDRITE =========================
for i in range(n):
    dendrite.append(Unit(diameters[int(math.floor(i*len(diameters)/n))], length, rm, rc, cm))

# calculate the effective distance
dd = 0
for j in range(n):
    dd += dendrite[-1-j].length/dendrite[-1-j].space_constant
    dendrite[-1-j].path_length = dd


# ======================== INPUT FUNCTIONS =======================
# single spike (r = input resistance, mag = magnitude)
def alpha(t, c, r):
    return r*c*c*t*np.exp(-c*t)


# complex spikes, mag = magnitude
def rep_alpha(t, a, n, r):
    c = np.zeros(len(t))
    if n > 0:
        for i in range(n):
            c += np.piecewise(t, [t <= i, t > i], [0, lambda t:alpha(t-i, a, r)])
        return c


# differentiate potential, c:current
def delta(m, t, a, x, n, r):
    return np.piecewise(t, [t <= m, t > m],
                        [0, lambda t:0.01*rep_alpha(t, a, n, r)*np.exp(-(t-m)-0.25*x*x/(t-m))/np.sqrt(t-m)])


# ======================== OUTPUT FUNCTION =======================
# Potential. beta(time_array, latency, alpha_constant, effective_distance,
# spike_quantity, input_resistance, magnitude)
def beta(t, moment, a, x, n, r, mag=1):
    v = np.zeros(len(t))
    for m in t:
        v += mag*delta(m, t-moment, a, x, n, r)
    return v

if len(injection) != 0:
    hillock.input_voltage = np.zeros(len(time))
    k = len(time)/injection.shape[1]
    x = np.nditer(injection, flags=['multi_index'])
    while not x.finished:
        if x[0] != 0:
            hillock.input_voltage += beta(time, x.multi_index[1], a, dendrite[x.multi_index[0]*10].path_length,
                                          1, dendrite[x.multi_index[0]*10].in_resistance, x[0])
        x.iternext()

# ============================ VISUALIZATION =====================
plt.plot(time, hillock.input_voltage)
plt.show()