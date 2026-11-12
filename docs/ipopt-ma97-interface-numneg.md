# IPOPT's MA97 interface reads an uninitialised value (and how to fix it)

IPOPT loads smf through its HSL MA97 interface (`linear_solver ma97` with `hsllib`). That interface has a bug
that makes IPOPT choose between two strategies by chance at the start of a solve. It is IPOPT's bug, not
smf's: the real HSL MA97 goes through the same code. Present in IPOPT 3.14.19 and on the `stable/3.14` and
`master` branches as of 2026-11-12. Reported upstream as [coin-or/Ipopt#870](https://github.com/coin-or/Ipopt/issues/870).

Patch: [`tools/ipopt/ma97-numneg.patch`](../tools/ipopt/ma97-numneg.patch) (two lines, against 3.14.19).

## What happens

`Ma97SolverInterface::MultiSolve` (`IpMa97SolverInterface.cpp`) stores the number of negative eigenvalues of
a factorization in `numneg_` only after the inertia check has passed:

```cpp
if( check_NegEVals && info.num_neg != numberOfNegEVals )
{
   ...
   return SYMSOLVER_WRONG_INERTIA;      // numneg_ not updated on this path
}
...
numneg_ = info.num_neg;
```

`numneg_` has no initialiser and is not set in the constructor or in `InitializeImpl`. But the caller reads it
right after a wrong-inertia return (`IpPDFullSpaceSolver.cpp`):

```cpp
else if( retval == SYMSOLVER_WRONG_INERTIA && augSysSolver_->NumberOfNegEVals() < numberOfEVals )
{
   // "Number of negative eigenvalues too small!": ask the solver for better quality first
```

So:

- **Before the first factorization with the right inertia of a solve**, `NumberOfNegEVals()` returns whatever
  the freshly allocated object held. If that happens to be smaller than the expected count, IPOPT raises the
  pivot tolerance and switches scaling on ("improve quality"); otherwise it perturbs the Hessian. Which one
  depends on heap contents, so on address randomisation and on everything the process allocated before.
- **After that**, it returns the count of the last *accepted* factorization, which equals the expected
  count, so the "too few negative eigenvalues" branch is never taken through this interface, even when the
  count really is too small.

The MA27 and MA57 interfaces set their count before the check. The MA77, MA86 and SPRAL interfaces have the
same ordering as MA97.

A solve is exposed when its first KKT factorization has the wrong inertia: an exact Hessian that is indefinite
at the start point, for example. With an L-BFGS Hessian the first factorization usually has the right inertia
and nothing shows.

## How it was seen

A 964-variable optimal-control problem with an exact Hessian, solved twice in one process through CasADi
(IPOPT 3.14.19 built from source on Ubuntu 22.04, GCC 11, smf as the MA97 library, one thread). The second
solve took 46 iterations in about half of the runs and 123 in the others, with the same binary and inputs.

| Check | Result |
|---|---|
| Address randomisation off (`setarch -R`) | 46 in 10 of 10 runs |
| MUMPS instead of the MA97 interface (a separate build: CasADi's own IPOPT and MUMPS) | 46 in 8 of 8 runs |
| `MALLOC_PERTURB_=1` (new memory filled with `0xFE`, a negative int) | 123 in 8 of 8 |
| `MALLOC_PERTURB_=170` (filled with `0x55`, a large int) | 46 in 8 of 8 |
| IPOPT printing the value it reads at the decision | `1763730792` in the first solve, `1` in the second; expected count 1484 |
| IPOPT with the patch, any of the fills above | 46 in 8 of 8 each |

Both outcomes are valid solutions (objectives 2e-6 apart); the cost is run time and reproducibility. IPOPT's
own log shows the fork: after the same first factorization (`info.num_neg = 1487, but numberOfNegEVals = 1484`)
one run prints `Solving system with delta_x=1.000000e-04` and the other
`Number of negative eigenvalues too small!`, although 1487 is not smaller than 1484.

A build of smf with `SMF_DETERMINISTIC=ON` gave 123 in 12 of 12 runs at first. That was chance: it changes
what the heap holds, not what IPOPT reads, and with the memory fills above it follows the fill like the
default build.

## The fix

Store the count before the inertia check, and give the member an initial value:

```cpp
numneg_ = info.num_neg;
if( check_NegEVals && info.num_neg != numberOfNegEVals )
```

With it IPOPT sees the real count on a wrong-inertia return, as it does with MA27 and MA57. That also turns
the "too few negative eigenvalues" branch back on for this interface: IPOPT then asks the MA97 library for
better quality (higher pivot tolerance, scaling) before it assumes a singular system.

Effect on the problem set this came from (28 scenarios in two variants of the problem, smf, unpatched against
patched IPOPT): every solve succeeds either way; all 56 L-BFGS solves are identical; of 56 exact-Hessian
solves 3 changed iteration count (39 to 48, 57 to 36, 44 to 47), which are the solves where the unpatched
run had taken the chance branch.

## What this means for results in this repository

- IPOPT runs with `linear_solver ma97`, with smf or with HSL MA97, on a stock IPOPT can differ from run to run
  in iteration count on problems that start with a wrong-inertia factorization. Boards in `docs/performance`
  were measured with stock IPOPT builds and were not re-measured with the patch.
- When a comparison between smf and MA27 or MA57 shows different IPOPT iteration paths, check this first:
  MA27 and MA57 do not have the bug, so part of such a difference can come from the interface and not from
  the factorization.
- For reproducible runs build IPOPT with the patch. Nothing in the smf plugin can work around it: the value
  is IPOPT's own member.
