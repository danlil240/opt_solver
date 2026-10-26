# Phase 0 exit report

Hard IPOPT corpus, single-thread baseline. Pressure columns come from the smf plugin trace.

| model | split | n | factorizations | corrections | corr frac | delays/fact | 0.01 n | qualified by | ma27 status (iters) | mumps status (iters) | smf status (iters) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pglib_opf_case118_ieee | tuning | 4830 | 27 | 0 | 0.000 | 648.44 | 48.30 | delays | Optimal Solution Found. (26) | Optimal Solution Found. (26) | Optimal Solution Found. (26) |
| pglib_opf_case300_ieee | heldout | 10955 | 37 | 0 | 0.000 | 1838.46 | 109.55 | delays | Optimal Solution Found. (36) | Optimal Solution Found. (36) | Optimal Solution Found. (36) |
| pglib_opf_case500_goc__api | tuning | 18889 | 53 | 1 | 0.019 | 2850.75 | 188.89 | delays | Optimal Solution Found. (51) | Optimal Solution Found. (51) | Optimal Solution Found. (51) |
| pglib_opf_case1354_pegase | heldout | 50759 | 46 | 0 | 0.000 | 8253.28 | 507.59 | delays | Optimal Solution Found. (44) | Optimal Solution Found. (44) | Optimal Solution Found. (44) |
| pglib_opf_case2000_goc__sad | heldout | 87432 | 75 | 18 | 0.240 | 15600.07 | 874.32 | both | Optimal Solution Found. (53) | Optimal Solution Found. (53) | Optimal Solution Found. (55) |
| pglib_opf_case2869_pegase | tuning | 115365 | 52 | 0 | 0.000 | 19985.81 | 1153.65 | delays | Optimal Solution Found. (50) | Optimal Solution Found. (50) | Optimal Solution Found. (50) |
| pglib_opf_case4661_sdet | tuning | 159871 | 60 | 0 | 0.000 | 30482.10 | 1598.71 | delays | Optimal Solution Found. (57) | Optimal Solution Found. (58) | Optimal Solution Found. (57) |
| pglib_opf_case9241_pegase | heldout | 400511 | 73 | 3 | 0.041 | 67536.66 | 4005.11 | delays | Optimal Solution Found. (68) | Optimal Solution Found. (68) | Optimal Solution Found. (68) |
| cops_glider_400 | tuning | 3613 | 3197 | 1890 | 0.591 | 38.55 | 36.13 | both | Optimal Solution Found. (1347) | Optimal Solution Found. (1372) | Optimal Solution Found. (1306) |
| cops_rocket_800 | heldout | 5609 | 32 | 4 | 0.125 | 38.38 | 56.09 | corrections | Optimal Solution Found. (27) | Optimal Solution Found. (27) | Optimal Solution Found. (27) |
| cops_catmix_800 | tuning | 15200 | 19 | 0 | 0.000 | 759.84 | 152.00 | delays | Optimal Solution Found. (18) | Optimal Solution Found. (18) | Optimal Solution Found. (18) |
| cops_chain_800 | tuning | 5609 | 4814 | 1814 | 0.377 | 36149.23 | 56.09 | both | Optimal Solution Found. (1475) | Optimal Solution Found. (1571) | Maximum Number of Iterations Exceeded. (3000) |
| cops_pinene_400 | tuning | 16005 | 13 | 0 | 0.000 | 4101.31 | 160.05 | delays | Optimal Solution Found. (12) | Optimal Solution Found. (12) | Optimal Solution Found. (12) |
| cops_steering_800 | tuning | 7213 | 23 | 7 | 0.304 | 1853.39 | 72.13 | both | Optimal Solution Found. (15) | Optimal Solution Found. (15) | Optimal Solution Found. (15) |
| cops_elec_200 | tuning | 800 | 744 | 439 | 0.590 | 199.59 | 8.00 | both | Optimal Solution Found. (157) | Optimal Solution Found. (154) | Optimal Solution Found. (304) |
| cops_polygon_200 | heldout | 40600 | 70 | 29 | 0.414 | 1.09 | 406.00 | corrections | Optimal Solution Found. (40) | Optimal Solution Found. (40) | Optimal Solution Found. (40) |

Qualification: inertia_corrections/factorizations >= 0.1 or delays_total/factorizations >= 0.01 * n; only tuning models count.

Qualifying tuning models (10): pglib_opf_case118_ieee, pglib_opf_case500_goc__api, pglib_opf_case2869_pegase, pglib_opf_case4661_sdet, cops_glider_400, cops_catmix_800, cops_chain_800, cops_pinene_400, cops_steering_800, cops_elec_200

Qualifying by inertia corrections only (4): cops_glider_400, cops_chain_800, cops_steering_800, cops_elec_200. delays_total counts delay events, not distinct columns, so the delay criterion is weak evidence; the verdict above uses the plan's either-criterion rule.

**Phase 0 verdict: PASS** (need >= 3 tuning models)
