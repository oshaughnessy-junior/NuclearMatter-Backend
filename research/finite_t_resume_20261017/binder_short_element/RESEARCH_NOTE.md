# Short transparent free-npe element: frozen private benchmark

This private research driver exercises the accepted staged free-matter/source
adapter without changing its eight-path candidate, prerequisites or public API.
The single element has fixed nB=1e−7fm−3, initial T2MeV/Ye.30, absent photons,
absent muons and empty escaping electron-flavor neutrino radiation. The INCLUDED
network is all three PR54 EC,PC,neutron-decay total-event channels. No scattering,
muonic/pair sources, strong matter interactions, radiative corrections or full
hydrodynamics are included. Material remains imposed local free FD/pair equilibrium.

## Predeclared execution

PRELOCK.json was written before any source outputs. It fixes the exact fine
channel controls, actual driver hash, all126 files from the frozen binder manifest,
duration2e−4s, explicit Euler in(T,Ye),2/4steps,maximum6 fresh complete source
groups/planned5,88s alarm per group, and ledger gates. The same newly executed
initial source was shared ONLY between the identical initial states after fresh
canonical identity checks. Every changed stage uses its own full physical source
calculation; no old rate data, fit or fixed-rate feedback is used. Each actual
source and RHS belongs to freshly canonicalized EOS roots and full provenance.
There is no public ODE API or persisted cache.

All FIVE planned unique groups completed in7.94–8.64s each, one numerical thread.
Usage was checked immediately before each group:96% used, same baseline account/
window/reset1792652325, below97% stop. No reset credits were consumed. Exact frozen
binder files were verified before and after each call. Full states, original six
forward/backward/net channel moments, controls/current, source identities, module
origins, process execution and RHS are saved in group_0..4.json. Each evolving rung
has a full stage-to-source mapping. There were no failed/budget-stopped calls or
retries. The unused sixth source allowance was not spent.

The instantaneous fixed-density equations are

```
Yedot = (RPC+RD−REC)/nB,
Cv*Tdot = −Qnu − e_Y*Yedot.
```

Cv and e_Y are the genuine matched fixed-number thermal/composition derivatives,
with rest-inclusive physical masses and actual pair-net electron neutrality.
The escaping neutrino energy and separate nue/antinue numbers use exactly the
same stage weights as the material updates. Endpoint EOS energy is freshly
computed, not reconstructed from the escaped-energy ledger. Material entropy is
checked separately using the thermodynamic entropy chain. No empty-radiation
entropy or full-system H-theorem is asserted.

## Measured results

17/17 predeclared checks pass; all actual time-discretization errors remain in
RESULTS.json rather than being projected away.

|Steps|Final T(MeV)|Final Ye|Escaped E(MeV fm−3)|[delta e+Eescape]/Eescape|
|---:|---:|---:|---:|---:|
|2|1.999409520470466|.3015699483091133|3.028968896900208e−9|2.508151525e−4|
|4|1.999409626300147|.3015684519688716|3.027953489378919e−9|1.253734751e−4|

Energy defects7.59711e−13/3.79625e−13MeVfm−3 halve on this refinement. Relative
to the initial rest-subtracted reference energy2.67962e−6MeVfm−3 they are
2.83515e−7/1.41671e−7. That reference subtracts nucleon and NET-electron rest
counts; its remaining energy retains e± pair rest energy and is not pure kinetic
energy. The primary gate uses the much smaller gross escaped-energy scale (2%).

Endpoint differences h−h/2 are delta T=−1.05830e−7MeV and delta Ye=1.49634e−6.
Ye changes by about.00157, and actual stage rates change: Rinitial7.864758e−7
falls to7.819795e−7fm−3s−1 at the last fine stage; Q changes1.515502e−5 to
1.512453e−5MeVfm−3s−1. Thus source coupling is exercised at changed states.
These TWO rungs alone do not determine a reliable asymptotic convergence order,
long-time trajectory accuracy or uniform error/domain certificate.

Electron Le residual divided by gross emitted number is below6.7e−15; actual
endpoint baryon/charge roots agree at~3e−14 relative. Individual source energy
ledgers are below9.3e−14 on their declared gross scales. Material entropy drops
by~.01294 per baryon, consistent with escape; entropy-chain discrepancies shrink
2.97e−5 to1.47e−5 relative to the integrated absolute material entropy source
(2% predeclared gate). ADDITIONAL_ACCOUNTING.json separately displays integrated
Cv*Tdot and e_Y*Yedot, baryon sum and absolute Le residual; it adds diagnostic
accounting, not retrospective pass gates.

## Review and limitations

Independent energy-variable/Heun/root evolution review remains pending. Fine
source controls inherit the selected adapter qualification; this run adds no
new independent phase-space convergence proof at every evolving state. The
initial source is executed fresh and exactly matches the accepted initial bath,
but the ordinary Euler benchmark is not a reference trajectory. Pressure work
vanishes here because nBdot=0. The two-rung source/energy/Le evidence is useful
for integration review, with modest short-time errors explicitly retained.

Run each command with the staged package on PYTHONPATH and
OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1 using the existing
junior_tools Python environment. element.py lock is intentionally non-overwriting;
then group0..4 are separate bounded processes, followed by element.py report.
For a new experiment copy the private driver into a fresh owned directory, create
a fresh prelock and recheck resources before each group. Existing completed
records deliberately reject overwrites. No production/paper/GitHub mutation,
dependency installation or changes to frozen sources were made.
