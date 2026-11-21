"""Per-configuration summary of the cuRobo comparison records in results/."""
import json
import pathlib
import statistics as st

R = pathlib.Path(__file__).resolve().parent / "results"


def load(name):
    p = R / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def row(label, recs, wall_key, scale=1.0):
    ok = [r for r in recs if r.get("success")]
    walls = [r[wall_key] * scale for r in recs]
    dur = [r["duration_s"] for r in ok]
    clr = [r.get("curobo_sphere_clearance") for r in ok if r.get("curobo_sphere_clearance") is not None]
    return (f"| {label} | {len(ok)}/{len(recs)} | {st.median(walls):.0f} | {min(walls):.0f}-{max(walls):.0f} | "
            f"{st.median(dur):.2f} | {min(dur):.2f}-{max(dur):.2f} | "
            f"{(min(clr) * 1000 if clr else float('nan')):.1f} |")


EVAL = {r["file"]: r["curobo_sphere_clearance"] for r in load("curobo_eval_ipopt.jsonl")}


def main():
    print("| configuration | success | median ms | range ms | median duration s | duration range s | worst sphere clearance mm |")
    print("|---|---|---|---|---|---|---|")
    print(row("cuRobo, sphere as mesh", load("curobo_mesh.jsonl"), "wall_ms_median"))
    print(row("cuRobo, sphere as bounding cube", load("curobo_cube.jsonl"), "wall_ms_median"))
    for f, lab in [("ipopt_kin32_cs_smf.jsonl", "IPOPT+smf, cuRobo spheres"),
                   ("ipopt_kin32_cs_ma27.jsonl", "IPOPT+MA27, cuRobo spheres"),
                   ("ipopt_kin32_cs_mumps.jsonl", "IPOPT+MUMPS, cuRobo spheres"),
                   ("ipopt_kin32_cs_smf_jit.jsonl", "IPOPT+smf, cuRobo spheres, compiled"),
                   ("ipopt_kin32.jsonl", "IPOPT+smf, arm7 points"),
                   ("ipopt_kin32_jit.jsonl", "IPOPT+smf, arm7 points, compiled"),
                   ("ipopt_kin32_ma27.jsonl", "IPOPT+MA27, arm7 points"),
                   ("ipopt_kin32_mumps.jsonl", "IPOPT+MUMPS, arm7 points"),
                   ("ipopt_dyn100_sjit.jsonl", "full arm7 (torque, RK4, N=100), stage-compiled")]:
        recs = load(f)
        if recs:
            for r in recs:
                r["success"] = r.get("status") == "Solve_Succeeded"
                if r.get("curobo_sphere_clearance") is None:
                    # arm7-point solutions: checked on the GPU afterwards (curobo_arm7.py --eval);
                    # every solver reached the same solution, saved from the smf run
                    key = f"traj_ipopt_{r['form']}_N{r['N']}_s{r['seed']}.npy"
                    r["curobo_sphere_clearance"] = EVAL.get(key)
            print(row(lab, recs, "wall_s", 1000.0))


if __name__ == "__main__":
    main()
