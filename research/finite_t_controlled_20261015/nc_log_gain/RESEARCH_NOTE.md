# Combined-log NC gain/loss at the excluded cold point

Research prototype,15October2026; independent review required. This **new** interface repairs a floating-point representation failure for one selected colder neutral-current redistribution experiment. The frozen continuum source and its excluded overflow evidence remain unchanged. Two newly measured coarse-grid failures also remain failed. No physical response, current, tail width or support is modified.

## Preserved failure and physical scope

The exact failed state is a free-NR FD neutron bathT=.08MeV,n=.0042fm^-3,M=939MeV,offset0, incidentE=19.6MeV, withFD radiation(Tnu=.4MeV,munu=5MeV). The accepted leading neutron NC response/current is reused unchanged. Its width-dominated outgoing-energy cutoff is118.27309753794682MeV.

AtE'=77.2MeV, the old integrand evaluatesexp[(E'−E)/T]=exp(720) before multiplying the tiny forward kernel. It throwsOverflowError although the forward, reverse and combined gain logarithms remain finite. The old source still reproduces that exception. `review_nc_continuum/overflow_log_evidence.json` and its failure reports are pinned and read-only. Their T=.08 extension remains excluded from the **old API**.

This leaf retains the existing free unpolarized NR neutron model, leading density/spin current, quadratic recoil, nuclear blocking, one-helicity conventions and externally prescribed radiation. Interactions, relativistic/current corrections, global double-energy energy exchange and evolution remain absent. It extends a numerical selected-point qualification, not a cold-state rectangle.

## Log-domain equations

The bareK(E→E',c) already includes outgoingE'². Writeomega=E−E' and occupationsfi=f(E),fo=f(E'). Withg(E) proportional toE²,

    log loss = logK_forward + logfi + log(1−fo),
    log gain_KMS = logK_forward − omega/T + logfo + log(1−fi),
    log gain_direct = logK_reverse + 2log(E'/E) + logfo + log(1−fi).

Only the final **combined** logarithms are exponentiated for optional ordinary outputs. No separately exponentiated reverse factor, floor, support cut or omitted channel is used. The direct path evaluates the reversed physical response; it does not reconstruct that response through KMS. FD log occupations use log_expit. The inherited packet distribution is handled as a logsum of its FD and Gaussian components.

For signed differences,log|exp(a)−exp(b)| is formed with expm1 using the larger logarithm as scale. ForFD radiation, pointwise net cancellation usesdelta=omega(1/Tnu−1/T) with stable large-positive/negative branches; the independent direct-reverse path instead obtains the difference from its two actual log coefficients. The response is unchanged. At equal computed logs the returned zero is algebraic **at working precision**; its flag is not a certificate resolving arbitrarily tiny departures from equilibrium.

Angularq quadrature uses the exactdc=q dq/(EE') Jacobian. Positive outgoing rate, transport, gain/loss, positive/negative energy-transfer and positive/negative net-source pieces are integrated separately by logsumexp over nodes and panels. The final signed energy/source moments retain sign, finite logarithmic magnitude when nonzero, optional ordinary value and explicit underflow/overflow flags. Algebraic empty/zero pieces useNone log with zero sign, not a fabricated negative-infinite physical rate. Raw panel logs are retained even when their ordinary contribution is unrepresentable.

## Selected results and convergence

At the exact failure witnessE'=77.2,c=−.3,loggain=−2443.55803566139; KMS/direct paths agree. Its ordinary value is unavailable and explicitly flagged underflow, while its finite logarithm remains returned. This is a representation repair, not evidence of physical support ending or of accurately known relative deep-tail rates.

The final selected cold reference uses outgoing-energy Gauss192 and angularq640, with the same physical panel construction. Its moments are:

| Moment | Value |
|---|---:|
| blocked out event rate [s^-1] |179082.058058|
| blocked angular transport [s^-1] |211504.579695|
| signed energy into bath [MeVs^-1] |328400.416664|
| absolute energy transfer [MeVs^-1] |328594.114624|
| gain [s^-1] |1.84258052710e-12|
| loss [s^-1] |2.51943690868e-11|
| netC(E) [s^-1] |−2.33517885597e-11|

All these integrated ordinary values are representable. They remain per-incident spectral quantities; they are not total fluid heating or cooling rates.

Energy/angular128/384→160/512→192/640 refinement passes the unchanged3e-5 gate; the largest comparison is5.84e-6, in the128-grid gain. The160→192 gain difference is approximately1.32e-7. At192/640, independent direct-reverse and KMS paths pass the same gate, and actually extending the absolute upper cutoff by1.5 leaves the moments unchanged on the1e-8 gate. This is a finite-cutoff test, not a rigorous infinite-tail bound.

The final values also agree with the **archived formerly independent q-outer/E-inner** cold reference within3e-5. That comparison has a genuinely different integration ordering, but this author now reads its frozen values; it is not a new independent review of the present interface. Fresh independent review remains required.

Warm(T2,n.002,E10,FD3/4) and nearLTE(FD2.00002/6) cases reproduce the accepted old continuum values within the same3e-5 gate. Log accumulation preserves the small nearLTE net source without subtracting large integrated rates. No rapid-relaxation inference is drawn from finite cold tails.

## Failures, inputs and reproducibility

The initial109-check study has **107pass/2fail** and remains `passed:false`. Its cold48/144 and64/192 gain-refinement errors against96/288 are9.14174e-5 and3.48105e-5, exceeding3e-5. The separate finer cold study adds35passing checks; it does not relabel those coarse failures. Input/representation tests add30passing checks, including actual subnormal behavior, nonfinite controls, finite signed-log differences and explicit structural zeros.

An initial refinement-driver syntax error occurred before physics; its source/error are preserved. A subsequent helper input-guard hardening rejects nonfinite logarithms even besideNone. The previous source/results are preserved, and the same physical studies were rerun on the final helper. No physical formula or acceptance tolerance changed. The final main and cold-refinement numerical runs take about10.4s and13s; their initial counterparts are retained separately. Every complete phase/run checkpoints raw evidence and a105s experiment guard is enforced.

Run `study.py`, then `refine_cold.py`, then `check_controls.py`. Qualified selected cold use should specify the recorded fine controls explicitly; the cheaper default controls are not a broad cold accuracy certificate. `CHECKPOINT.json` distinguishes accepted fine evidence from the two preserved coarse failures and pins frozen upstream/failure evidence. No production, git, papers repository or other contributor's file was changed.
