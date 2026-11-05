"""Quadrotor dynamics (12-state rigid body, 4 rotor thrusts) shared by the
quadrotor problems.

State: position p (3), velocity v (3, world frame), ZYX Euler angles
(roll phi, pitch theta, yaw psi), body rates w (3). Input: the four rotor
thrusts (N), rotors in an X layout at arm length L. Linear drag on v.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

MASS = 1.0
GRAV = 9.81
ARM = 0.17
KAPPA = 0.016          # yaw torque per thrust (m)
INERTIA = np.array([0.0082, 0.0082, 0.0149])
DRAG = 0.1
U_MAX = 6.0            # per rotor; hover is m g / 4 = 2.45 N
NX, NU = 12, 4
HOVER = MASS * GRAV / 4


def rotation(phi, theta, psi):
    cph, sph = ca.cos(phi), ca.sin(phi)
    cth, sth = ca.cos(theta), ca.sin(theta)
    cps, sps = ca.cos(psi), ca.sin(psi)
    return ca.vertcat(
        ca.horzcat(cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps),
        ca.horzcat(cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps),
        ca.horzcat(-sth, sph * cth, cph * cth))


def dynamics(x, u):
    v = x[3:6]
    phi, theta, psi = x[6], x[7], x[8]
    w = x[9:12]
    thrust = u[0] + u[1] + u[2] + u[3]
    # X layout: rotors 0..3 at +-45 deg, spinning alternately.
    a = ARM / np.sqrt(2.0)
    tau = ca.vertcat(a * (u[0] - u[1] - u[2] + u[3]),
                     a * (-u[0] - u[1] + u[2] + u[3]),
                     KAPPA * (u[0] - u[1] + u[2] - u[3]))
    R = rotation(phi, theta, psi)
    acc = R[:, 2] * thrust / MASS - ca.vertcat(0, 0, GRAV) - DRAG * v / MASS
    # Euler-angle rates from body rates.
    tth = ca.tan(theta)
    cph, sph, cth = ca.cos(phi), ca.sin(phi), ca.cos(theta)
    eta_dot = ca.vertcat(w[0] + sph * tth * w[1] + cph * tth * w[2],
                         cph * w[1] - sph * w[2],
                         (sph * w[1] + cph * w[2]) / cth)
    J = INERTIA
    w_dot = ca.vertcat((tau[0] - (J[2] - J[1]) * w[1] * w[2]) / J[0],
                       (tau[1] - (J[0] - J[2]) * w[2] * w[0]) / J[1],
                       (tau[2] - (J[1] - J[0]) * w[0] * w[1]) / J[2])
    return ca.vertcat(v, acc, eta_dot, w_dot)


# State bounds: tilt below ~57 deg keeps the Euler map regular.
X_LB = np.array([-1e3, -1e3, 0.2, -12, -12, -6, -1.0, -1.0, -np.pi, -6, -6, -3])
X_UB = np.array([1e3, 1e3, 6.0, 12, 12, 6, 1.0, 1.0, np.pi, 6, 6, 3])
