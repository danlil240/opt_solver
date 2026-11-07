"""cuRobo plan_cspace on the arm7 scenes (same start, goal, sphere per seed).

Runs in cuRobo's own environment (Linux, CUDA). Times each plan (wall,
CUDA-synchronised, after warmup), records cuRobo's own success and motion
duration, and checks the interpolated trajectory with the arm7 problem's
constraints (arm7_model.check) and with cuRobo's own collision spheres
(clearance to the obstacle sphere and to the table top).

  python curobo_arm7.py --seeds 10 --reps 5 [--obst mesh|cube] [--graph 1]
  python curobo_arm7.py --eval 'traj_ipopt_*.npy'   # check IPOPT trajectories

--obst mesh: the sphere as a triangle mesh (cuRobo's planner does not load
             sphere primitives: SceneCfg spheres are dropped silently unless
             converted). cube: its bounding cube (conservative).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm7_model as am  # noqa: E402
from curobo._src.geom.types import SceneCfg  # noqa: E402
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
from curobo.types import JointState  # noqa: E402

TABLE_TOP = -0.025


def scene_cfg(r, mode):
    d = {"cuboid": {"table": {"dims": [1.4, 1.4, 0.05], "pose": [0.0, 0.0, -0.05, 1, 0, 0, 0]}},
         "sphere": {"obst": {"radius": float(r), "pose": [*am.OBST.tolist(), 1, 0, 0, 0]}}}
    if mode == "cube":
        return SceneCfg.create(d).get_obb_world()
    return SceneCfg.create_collision_support_world(SceneCfg.create(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--graph", type=int, default=1, help="attempt index where graph seeding starts")
    ap.add_argument("--timeopt", type=int, default=1)
    ap.add_argument("--obst", default="mesh", choices=["mesh", "cube"])
    ap.add_argument("--eval", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default="curobo_arm7.jsonl")
    args = ap.parse_args()

    _, _, r0 = am.scene(0)
    cfg = MotionPlannerCfg.create(robot="franka.yml", scene_model=scene_cfg(r0, args.obst),
                                  collision_cache={"obb": 4, "mesh": 4},
                                  interpolation_dt=0.005, enable_timeopt=bool(args.timeopt),
                                  interpolation_buffer_size=4000)
    planner = MotionPlanner(cfg)
    names = planner.joint_names
    dev = planner.default_joint_state.position.device

    def js(q):
        return JointState.from_position(
            torch.tensor(np.asarray(q), dtype=torch.float32, device=dev).view(1, -1), joint_names=names)

    def sphere_clearance(Q, r):
        ks = planner.compute_kinematics(JointState.from_position(
            torch.tensor(Q, dtype=torch.float32, device=dev), joint_names=names))
        s = ks.robot_spheres.reshape(Q.shape[0], -1, 4).cpu().double().numpy()
        s = s[:, s[0, :, 3] > 0]  # inactive spheres have negative radius
        obst = np.min(np.linalg.norm(s[..., :3] - am.OBST, axis=-1) - s[..., 3] - r)
        table = np.min(s[..., 2] - s[..., 3] - TABLE_TOP)
        return float(obst), float(table)

    out = open(args.out, "a")
    if args.eval:
        for f in sorted(glob.glob(args.eval)):
            seed = int(f.rsplit("_s", 1)[1].split(".")[0])
            _, _, r = am.scene(seed)
            o, t = sphere_clearance(np.load(f), r)
            rec = {"file": os.path.basename(f), "seed": seed,
                   "curobo_sphere_clearance": o, "curobo_table_clearance": t}
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
        return

    t = time.perf_counter()
    planner.warmup(enable_graph=True)
    print("warmup s", time.perf_counter() - t, flush=True)

    for seed in range(args.seeds):
        qs, qg, r = am.scene(seed)
        planner.scene_collision_checker.clear_cache()
        planner.update_world(scene_cfg(r, args.obst))
        start, goal = js(qs), js(qg)
        walls, solves, res = [], [], None
        for rep in range(args.reps):
            planner.reset_seed()
            torch.cuda.synchronize()
            t = time.perf_counter()
            res = planner.plan_cspace(goal, start, max_attempts=20, enable_graph_attempt=args.graph)
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t)
            solves.append(float(res.solve_time) if res is not None else float("nan"))
            print(f"  seed {seed} rep {rep} wall {1e3 * walls[-1]:.1f} ms  solve {1e3 * solves[-1]:.1f} "
                  f"attempts {planner.last_planning_attempts}", flush=True)
        rec = {"tag": args.tag, "obst": args.obst, "seed": seed, "obst_r": r,
               "wall_ms": [1e3 * w for w in walls], "wall_ms_median": 1e3 * float(np.median(walls)),
               "solve_ms": [1e3 * s for s in solves],
               "success": bool(res is not None and res.success.any()),
               "attempts": planner.last_planning_attempts,
               "source": getattr(res, "trajectory_source", None) if res is not None else None,
               "total_ms": 1e3 * float(res.total_time) if res is not None else None,
               "graph_ms": 1e3 * planner.last_graph_time, "timeopt": args.timeopt,
               "graph_from": args.graph}
        if rec["success"]:
            tr = res.get_interpolated_plan()
            P = tr.position.reshape(-1, tr.position.shape[-2], tr.position.shape[-1])[0]
            dt = float(tr.dt.reshape(-1)[0])
            Q = P.cpu().double().numpy()[:, :7]
            QD = np.gradient(Q, dt, axis=0)
            QDD = np.gradient(QD, dt, axis=0)
            rec.update({"duration_s": dt * (Q.shape[0] - 1), "dt": dt, "n": int(Q.shape[0]),
                        "start_err": float(np.max(np.abs(Q[0] - qs))),
                        "goal_err": float(np.max(np.abs(Q[-1] - qg))),
                        "path_len_rad": float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1))),
                        "max_acc": float(np.max(np.abs(QDD[2:-2])))})
            rec.update({"arm7_" + k: v for k, v in am.check(Q[::2], QD[::2], QDD[::2], r).items()})
            o, tb = sphere_clearance(Q, r)
            rec["curobo_sphere_clearance"], rec["curobo_table_clearance"] = o, tb
            np.save(os.path.join(args.save_dir, f"traj_curobo{args.tag}_s{seed}.npy"), Q)
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
