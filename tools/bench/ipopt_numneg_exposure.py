"""Which IPOPT runs were exposed to the numneg bug of IPOPT's MA97 interface (docs/ipopt-ma97-interface-numneg.md)?

Usage: python tools/bench/ipopt_numneg_exposure.py <directory of IPOPT logs>

Each log must hold IPOPT's output and the plugin trace (SMF_PLUGIN_STATS=2: one "[smf_ma97] factor" line per
factorization with its inertia). m is the expected number of negative eigenvalues (constraints). Columns:
  fact    factorizations of the main KKT system; other = systems of another size (restoration phase)
  win     factorizations before the first one with the right inertia: IPOPT reads an uninitialised count here
          (low / high: fewer / more negative eigenvalues than m)
  l.low   later factorizations with fewer than m negative eigenvalues (or zeros): IPOPT reads a stale count equal
          to m and perturbs instead of first asking for better quality
  l.high  later factorizations with more than m: the stale count gives the same decision as the true one
  q1e-6   factorizations with scaling on and pivot tolerance 1e-6 (the "better quality" request)
"""
import re, sys, glob, os
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, '*.log'))):
    s = open(f, encoding='utf-8', errors='replace').read()
    def num(label):
        m = re.search(label + r'\.*:\s*(\d+)', s)
        return int(m.group(1)) if m else None
    nv, ne, ni = num('Total number of variables'), num('Total number of equality constraints'), num('Total number of inequality constraints')
    fac = [(int(a), int(b), int(c)) for a, b, c in re.findall(r'\[smf_ma97\] factor \d+: .*?pos=(\d+) neg=(\d+) zero=(\d+)', s)]
    scale_out = len(re.findall(r'scaling=1 scale=out u=1\.0e-06', s))
    if nv is None or not fac:
        rows.append((os.path.basename(f)[:-4], 'no data', nv, len(fac))); continue
    m = ne + ni; dim = nv + ni + m
    main = [(i, p, n, z) for i, (p, n, z) in enumerate(fac) if p + n + z == dim]
    other = len(fac) - len(main)
    first_good = next((k for k, (i, p, n, z) in enumerate(main) if n == m and z == 0), None)
    window = main[:first_good] if first_good is not None else main
    w_low = sum(1 for (i, p, n, z) in window if n < m or z)
    w_high = sum(1 for (i, p, n, z) in window if n > m and not z)
    later = main[first_good + 1:] if first_good is not None else []
    l_low = sum(1 for (i, p, n, z) in later if n < m or z)     # stale value (= m) hides "too few negatives"
    l_high = sum(1 for (i, p, n, z) in later if n > m and not z)
    it = re.search(r'Number of Iterations\.*:\s*(\d+)', s)
    rows.append((os.path.basename(f)[:-4], m, len(main), other, len(window), w_low, w_high, l_low, l_high, scale_out, it.group(1) if it else '-'))
print('%-16s %7s %6s %6s | %5s %4s %5s | %5s %5s | %5s %5s' % ('problem', 'm', 'fact', 'other', 'win', 'low', 'high', 'l.low', 'l.high', 'q1e-6', 'iter'))
for r in rows:
    if r[1] == 'no data': print('%-16s no data (nv=%s, factor lines=%s)' % (r[0], r[2], r[3])); continue
    print('%-16s %7d %6d %6d | %5d %4d %5d | %5d %5d | %5d %5s' % r)
