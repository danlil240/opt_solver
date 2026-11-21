"""Fit cuRobo's Franka collision spheres to the arm7 DH frames.

For random q, cuRobo's robot_spheres (world) are regressed on each DH frame
(base, joints 1..7): the frame whose constant local offset reproduces the
sphere centre (residual ~0) is its parent. Writes curobo_spheres.json:
[{frame, offset[3], radius}] for the arm7 CasADi model."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm7_model as am
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState


def dh_frames(q):
    T = np.eye(4)
    out = [T.copy()]
    for i, (a, d, al) in enumerate(am.DH):
        T = T @ am._tf(a, d, al, q[i])
        out.append(T.copy())
    return out  # 8 frames: base, after joint 1..7


def main():
    pl = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml", scene_model=None, use_cuda_graph=False))
    dev = pl.default_joint_state.position.device
    rng = np.random.default_rng(0)
    Q = rng.uniform(am.Q_LO, am.Q_LIM, size=(64, 7))
    ks = pl.compute_kinematics(JointState.from_position(torch.tensor(Q, dtype=torch.float32, device=dev),
                                                        joint_names=pl.joint_names))
    S = ks.robot_spheres.reshape(Q.shape[0], -1, 4).cpu().double().numpy()
    F = [dh_frames(q) for q in Q]
    res = []
    for s in range(S.shape[1]):
        rad = S[0, s, 3]
        if rad <= 0:
            continue
        best = None
        for f in range(8):
            # c = R p + t  ->  R^T (c - t) = p, averaged
            loc = np.array([F[k][f][:3, :3].T @ (S[k, s, :3] - F[k][f][:3, 3]) for k in range(len(Q))])
            p = loc.mean(0)
            err = np.max(np.abs(loc - p))
            if best is None or err < best[0]:
                best = (err, f, p)
        res.append({"frame": best[1], "offset": best[2].tolist(), "radius": float(rad), "fit_err": float(best[0])})
    print("spheres", len(res), "max fit err", max(r["fit_err"] for r in res))
    print("frames", np.bincount([r["frame"] for r in res], minlength=8))
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_spheres.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
