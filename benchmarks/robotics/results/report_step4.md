# Robotics suite, first measurements (2026-11-06)

## Setup

- **Machine:** Ryzen dev PC, 1 thread, one solve per process at below-normal priority, seed 0.
- **Builds:** smf main d6fd34a through the MA97 plugin; MUMPS bundled with CasADi 3.8.2; MA27, MA57 and MA97 from CoinHSL 2024.5.15.
- **Functions:** CasADi's NLP functions run in its virtual machine (JIT off), so the function-evaluation share of wall time is overstated.
- **Linear-solver time:** IPOPT's own timers (symbolic + numerical factorization + back solves). MUMPS is left out of linear-time comparisons: IPOPT does not split its factorization out.
- **Per-cell table:** summary_t1.md.

## Results

**smf / best rival, geomean over the 4 sizes:**

| problem | wall | linear time per iteration |
|---|---:|---:|
| quad_forest | 1.02 | 0.90 |
| drone_swap | 1.12 | 1.02 (N=50/100: 1.28/1.61; N=200/400: 0.72/0.73) |
| car_track | 0.98 | 0.93 |
| truck_trailer | 2.04 | 1.48 |
| arm7 | 1.00 | 0.67 |

**KKT replay** (captured IPOPT sequences, 12900K, per factorization at 1T):
- smf is faster than MA27 on arm (2.5x), quad (1.5x), swap (1.75x; 3x at 8T) and car (1.17x).
- smf is 1.64x slower than MA27 on truck.

## Where smf loses

1. **Tiny fronts (truck_trailer, inequality-heavy KKTs).** The 17k inequality slacks and their multipliers become 1-4 row leaf fronts: 17.7k supernodes for n = 36.6k.
   - smf factorizes 1.36x (live) to 1.64x (replay) slower than MA27.
   - The profile is spread over per-front fixed costs: assembly scatter, D-entry push_back, small-front buffer setup, value mapping and the pivot bookkeeping.
2. **Iteration-path lottery (truck N=100: 1794 vs 590 iterations; swap N=100: 36 vs 26).**
   - Not an inertia problem: smf reports no static pivots, no inexact-inertia (flag 7), no zero pivots and no aborts.
   - smf and MA57 both depart from MA97 at the same badly conditioned step (truck iteration ~148, ||d|| 2.7e4), agreeing to ~7 digits before it.
   - MA57 happens to rejoin MA97's optimum and smf does not. Judging it needs several seeds.
3. **Analysis on inequality-heavy KKTs.** smf 33-39 ms vs MA27 5 ms on truck: the auto matching trial runs, then adopts a matching ordering with 15% more flops.
   - IPOPT analyses once per solve, so this is under 1% of a cold solve.
   - It would matter for warm replanning, which re-analyses per call.
4. **Small fixed per-call costs** at N=50 (car 1.10, swap 1.28 per iteration).

**Linear-solver share of wall time** with interpreted functions:

| problem | share |
|---|---|
| arm | 2-5% |
| quad | 26-37% |
| swap | 30-49% |
| car / truck | 45-64% |
