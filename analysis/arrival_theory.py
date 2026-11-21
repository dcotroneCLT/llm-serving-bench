"""Theoretical interarrival CoV of the client's arrival processes (R7).

The client (client/benchmark.py) generates (any other mode name falls back to
poisson in make_arrival_process, and is treated the same way here):
  * poisson: exponential gaps at `rate`, CoV 1;
  * constant: a fixed gap 1/rate, theoretical CoV 0. The realized gaps are
    dispatch timestamps, so event-loop jitter of rms sigma adds variance
    2*sigma^2 (successive gaps share an endpoint): the realized CoV is
    ~ sqrt(2)*sigma*rate. The band for constant runs is that jitter allowance
    with sigma = JITTER_RMS_S; the sampling band of the other modes does not
    apply (it is identically 0 at CoV 0). A poisson generator substituted for
    constant fails whenever sqrt(2)*JITTER_RMS_S*rate < 1, i.e. below ~14 rps;
  * bursty: BurstyArrival, an on/off modulated Poisson process (interrupted
    Poisson process, IPP). Arrivals at rate*b only while ON; ON and OFF
    sojourns are exponential with means on_s and on_s*(b-1), so the ON fraction
    is 1/b and the long-run rate is `rate`. Because an arrival always leaves
    the process ON and sojourns are memoryless, the IPP is a RENEWAL process:
    gaps are i.i.d. with a 2-phase phase-type (hyperexponential) law,
    PH(pi=[1,0], T=[[-(lam+alpha), alpha], [beta, -beta]]), with lam = rate*b,
    alpha = 1/on_s, beta = 1/(on_s*(b-1)). Raw moments E[X^k] = k! pi (-T)^-k 1.

The SCV is 1 + 2*lam*alpha/(alpha+beta)^2: at a low absolute rate a burst holds
less than one request and the process is nearly Poisson. That is why a fixed
CoV threshold misjudges correctly generated low-rate bursty runs (Dynamo p11:
0.0477 rps, CoV 1.46 against a theoretical 1.44).

The check compares the REALIZED sample CoV with the theoretical one for the
RECORDED rate / burst_factor / burst_on_seconds. With i.i.d. gaps the sample CoV
has the delta-method variance
    Var(c_hat) = [s^4/m^4 + (mu4 - s^4)/(4 s^2 m^2) - mu3/m^3] / n
(m mean, s^2 variance, mu3 / mu4 central moments). Tolerance band:
    |c_realized - c_theory| <= max(Z_BAND * SE, REL_BAND * c_theory)
Z_BAND = 4 bounds sampling noise; REL_BAND = 3% absorbs event-loop timing jitter
that dominates on very long, very fast runs (millions of gaps, SE ~ 1e-3).
On the 19 completed Dynamo runs the realized CoV sits within |z| <= 2.7 of the
theory (all within 2.2%); a Poisson generator substituted for bursty at p11's
rate lands at z ~ -22. When the theoretical bursty CoV is itself inside the
Poisson band the run cannot discriminate the generator: verdict low_power (a
note, not a failure).

Stdlib only (the validator imports it).
"""

from __future__ import annotations

import math
from typing import Optional

Z_BAND = 4.0
REL_BAND = 0.03
JITTER_RMS_S = 0.05      # dispatch-timestamp jitter allowance for constant arrivals
MODES = ("poisson", "constant", "bursty")


def normalize_mode(mode: Optional[str]) -> str:
    """The mode the client actually ran (make_arrival_process: unknown -> poisson)."""
    m = (mode or "poisson").lower()
    return m if m in MODES else "poisson"


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def interarrival_moments(mode: str, rate: float, burst_factor: float = 4.0,
                         on_seconds: float = 10.0) -> list[float]:
    """Raw moments E[X], E[X^2], E[X^3], E[X^4] of the gap, mirroring the
    client's clamps (burst_factor >= 1, on_seconds >= 1e-6)."""
    if not (rate > 0 and math.isfinite(rate)):
        raise ValueError(f"invalid rate {rate!r}")
    mode = normalize_mode(mode)
    if mode == "constant":
        return [rate ** -k for k in (1, 2, 3, 4)]
    b = max(1.0, float(burst_factor))
    if mode != "bursty" or b == 1.0:
        return [math.factorial(k) / rate ** k for k in (1, 2, 3, 4)]
    on = max(1e-6, float(on_seconds))
    lam, alpha, beta = rate * b, 1.0 / on, 1.0 / (on * (b - 1.0))
    det = lam * beta                     # det(-T)
    inv = [[beta / det, alpha / det],    # (-T)^-1
           [beta / det, (lam + alpha) / det]]
    power = [[1.0, 0.0], [0.0, 1.0]]
    moments = []
    for k in (1, 2, 3, 4):
        power = _matmul(power, inv)
        moments.append(math.factorial(k) * (power[0][0] + power[0][1]))
    return moments


def theoretical_cv(mode: str, rate: float, burst_factor: float = 4.0,
                   on_seconds: float = 10.0, n_gaps: Optional[int] = None) -> tuple[float, float]:
    """(theoretical CoV, standard error of the sample CoV over n_gaps gaps);
    SE is nan when n_gaps is not given."""
    m1, m2, m3, m4 = interarrival_moments(mode, rate, burst_factor, on_seconds)
    var = m2 - m1 ** 2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    if var <= 1e-12 * m1 ** 2:          # constant: degenerate law, no sampling noise
        return 0.0, (0.0 if n_gaps and n_gaps >= 2 else float("nan"))
    cv = math.sqrt(var) / m1
    if not n_gaps or n_gaps < 2:
        return cv, float("nan")
    v_hat = (var ** 2 / m1 ** 4 + (mu4 - var ** 2) / (4 * var * m1 ** 2) - mu3 / m1 ** 3) / n_gaps
    return cv, math.sqrt(max(v_hat, 0.0))


def band(cv: float, se: float, mode: str = "poisson", rate: float = 0.0) -> float:
    if normalize_mode(mode) == "constant":
        return math.sqrt(2.0) * JITTER_RMS_S * rate
    return max(Z_BAND * se, REL_BAND * cv)


def check_realized_cv(arrival: dict, config: Optional[dict] = None) -> tuple[str, str]:
    """Verdict on a run's client/arrival_stats.json against the theory for its
    RECORDED parameters: ('pass' | 'fail' | 'low_power', message). Missing or
    non-finite evidence fails (the generator cannot be verified)."""
    config = config or {}
    mode = normalize_mode(arrival.get("arrival_mode"))
    try:
        rate = float(arrival.get("target_rate_rps", config.get("target_rate_rps")))
        n = int(arrival["realized_count"]) - 1
        cv_real = float(arrival["interarrival_cv"])
        bf = float(arrival.get("burst_factor", config.get("burst_factor", 4.0)))
        on = float(arrival.get("burst_on_seconds", config.get("burst_on_seconds", 10.0)))
        if not (math.isfinite(cv_real) and n >= 2):
            raise ValueError("non-finite CoV or too few arrivals")
        cv_th, se = theoretical_cv(mode, rate, bf, on, n)
    except (KeyError, TypeError, ValueError) as exc:
        return "fail", f"arrival evidence incomplete for the CoV check ({exc})"
    tol = band(cv_th, se, mode, rate)
    if mode == "constant":
        detail = f"mode=constant, rate={rate:g}, n={n}, jitter allowance {JITTER_RMS_S:g}s rms"
    else:
        z = (cv_real - cv_th) / se if se > 0 else float("inf")
        detail = (f"mode={mode}, rate={rate:g}, burst_factor={bf:g}, on={on:g}s, "
                  f"n={n}, z={z:+.2f}")
    msg = f"realized CoV {cv_real:.3f} vs theoretical {cv_th:.3f} +/- {tol:.3f} ({detail})"
    if abs(cv_real - cv_th) > tol:
        return "fail", msg
    if mode == "bursty":
        poisson_cv, poisson_se = theoretical_cv("poisson", rate, n_gaps=n)
        if abs(cv_th - poisson_cv) <= band(poisson_cv, poisson_se):
            return "low_power", msg + "; indistinguishable from Poisson at this rate"
    return "pass", msg
