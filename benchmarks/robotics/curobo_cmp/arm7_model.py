"""NumPy copy of benchmarks/robotics/p_arm7.py: the scene (start, goal, obstacle
per seed), the link points, and the point-mass torque model. Used to check any
trajectory (cuRobo's or IPOPT's) with the arm7 problem's own constraints."""
import numpy as np

DH = [(0, 0.333, 0), (0, 0, -np.pi / 2), (0, 0.316, np.pi / 2), (0.0825, 0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0, 0, np.pi / 2), (0.088, 0, np.pi / 2)]
FLANGE_D = 0.107 + 0.1034
MASS = np.array([4.97, 0.65, 3.23, 3.59, 1.23, 1.67, 0.74 + 0.73])
Q_LIM = np.array([2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973])
Q_LO = -Q_LIM.copy()
Q_LO[3], Q_LIM[3] = -3.0718, -0.0698
Q_LO[5], Q_LIM[5] = -0.0175, 3.7525
QD_LIM = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
TAU_LIM = np.array([87, 87, 87, 87, 12, 12, 12.0])
ROTOR = np.array([0.6, 0.6, 0.5, 0.5, 0.15, 0.15, 0.1])
FRICTION = 0.5
OBST = np.array([0.45, 0.0, 0.45])


def scene(seed):
    rng = np.random.default_rng(seed)
    q_start = np.array([-1.2, 0.3, 0.0, -1.8, 0.0, 2.0, 0.8]) + rng.uniform(-0.1, 0.1, 7)
    q_goal = np.array([1.2, 0.3, 0.0, -1.8, 0.0, 2.0, 0.8]) + rng.uniform(-0.1, 0.1, 7)
    obst_r = 0.18 + rng.uniform(0, 0.04)
    return q_start, q_goal, obst_r


def _tf(a, d, al, q):
    ca, sa, cq, sq = np.cos(al), np.sin(al), np.cos(q), np.sin(q)
    return np.array([[cq, -sq, 0, a], [sq * ca, cq * ca, -sa, -sa * d],
                     [sq * sa, cq * sa, ca, ca * d], [0, 0, 0, 1]])


def frames(q):
    T = np.eye(4)
    origins, axes = [np.zeros(3)], []
    for i, (a, d, al) in enumerate(DH):
        T = T @ _tf(a, d, al, q[i])
        origins.append(T[:3, 3].copy())
        axes.append(T[:3, 2].copy())
    Tf = T.copy()
    Tf[:3, 3] = T[:3, 3] + FLANGE_D * T[:3, 2]
    origins.append(Tf[:3, 3])
    return origins, axes  # origins[0] is the base (CasADi's frames() omits it)


def points(q):
    o, _ = frames(q)
    o = o[1:]
    pts = []
    for i in range(1, len(o)):
        pts.append(o[i])
        pts.append(0.5 * (o[i] + o[i - 1]))
    return np.array(pts)[2:]  # the problem skips the two base points


def potential(q):
    o, _ = frames(q)
    return sum(MASS[i] * 9.81 * o[i + 2][2] for i in range(7))


def torque(q, qd, qdd):
    """tau = D(q) qdd + grad V + friction qd, the arm7 model."""
    e = 1e-6
    grav = np.array([(potential(q + e * np.eye(7)[i]) - potential(q - e * np.eye(7)[i])) / (2 * e)
                     for i in range(7)])
    o, z = frames(q)
    o = o[1:]
    diag = np.zeros(7)
    for i in range(7):
        m = ROTOR[i]
        for j in range(i, 7):
            r = o[j + 1] - o[i]
            perp = r - np.dot(r, z[i]) * z[i]
            m += MASS[j] * perp @ perp
        diag[i] = m
    return diag * qdd + grav + FRICTION * qd


def check(Q, QD, QDD, obst_r):
    """Q, QD, QDD: (T, 7) samples. Returns the arm7 constraint margins."""
    clear = min(float(np.min(np.linalg.norm(points(q) - OBST, axis=1))) - obst_r for q in Q)
    zmin = min(float(np.min(points(q)[:, 2])) for q in Q)
    taur = max(float(np.max(np.abs(torque(q, qd, qdd)) / TAU_LIM)) for q, qd, qdd in zip(Q, QD, QDD))
    return {"clearance": clear, "clear_ok": clear >= 0.06 - 1e-4, "min_height": zmin,
            "height_ok": zmin >= 0.05 - 1e-4, "max_tau_ratio": taur,
            "max_qd_ratio": float(np.max(np.abs(QD) / QD_LIM)),
            "q_ok": bool(np.all(Q >= Q_LO - 1e-4) and np.all(Q <= Q_LIM + 1e-4))}


# cuRobo's Franka collision spheres, fitted to the DH frames above
# (dump_spheres.py; fit error < 1e-6 m): frame 0 is the base, i the frame
# after joint i.
TABLE_TOP = -0.025


def load_spheres(path=None):
    import json, os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_spheres.json")
    return json.load(open(path))


def dh_frames(q):
    T = np.eye(4)
    out = [T.copy()]
    for i, (a, d, al) in enumerate(DH):
        T = T @ _tf(a, d, al, q[i])
        out.append(T.copy())
    return out


def sphere_check(Q, obst_r, spheres=None):
    """Clearance of cuRobo's robot spheres to the obstacle sphere and the table."""
    spheres = spheres or load_spheres()
    obst, table = np.inf, np.inf
    for q in Q:
        F = dh_frames(q)
        for s in spheres:
            c = F[s["frame"]][:3, :3] @ np.array(s["offset"]) + F[s["frame"]][:3, 3]
            obst = min(obst, float(np.linalg.norm(c - OBST)) - s["radius"] - obst_r)
            table = min(table, float(c[2]) - s["radius"] - TABLE_TOP)
    return {"sphere_clearance": obst, "table_clearance": table}
