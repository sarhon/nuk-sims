"""
Point Kinetics Equation (PKE) Simulation — CLI
===============================================

The reactor neutron population and delayed-neutron precursors are
governed by this system of 7 coupled ODEs:

  NEUTRON DENSITY
  ---------------
  dn/dt = [(ρ(t) - β) / Λ] · n(t)  +  Σᵢ λᵢ · Cᵢ(t)
              ↑ net prompt production     ↑ delayed neutron source

  PRECURSOR CONCENTRATIONS  (i = 1 … 6)
  ----------------------------------------
  dCᵢ/dt = (βᵢ / Λ) · n(t)  −  λᵢ · Cᵢ(t)
             ↑ precursor production  ↑ precursor decay

Variables
---------
  n(t)    neutron density, normalized so n(0) = 1
  Cᵢ(t)  precursor group i concentration
  ρ(t)   reactivity (dimensionless)
  β      total delayed neutron fraction  = Σ βᵢ
  βᵢ     delayed neutron fraction for group i
  Λ      prompt neutron generation time [s]
  λᵢ     decay constant for precursor group i [s⁻¹]

Reactivity units
----------------
  dimensionless ρ  ↔  dollars $  =  ρ / β
  dollars $        ↔  cents  ¢   =  100 · (ρ / β)

  Prompt supercritical threshold:  ρ = β  →  1 $  →  100 ¢

Steady-state initial conditions (ρ = 0)
----------------------------------------
  n(0)  = 1
  Cᵢ(0) = βᵢ / (λᵢ · Λ)     (set dCᵢ/dt = 0, solve for Cᵢ)

Usage
-----
  uv run python pke/pke_sim.py --mode step --rho 50 --unit cents --t0 1 --t-end 30
  uv run python pke/pke_sim.py --help
"""

from __future__ import annotations

import argparse
import csv
import sys
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("Agg")  # non-interactive backend — saves files, no display needed

# ─────────────────────────────────────────────────────────────────────────────
# Physics constants — Keepin 6-group data for U-235 thermal fission (1965)
# ─────────────────────────────────────────────────────────────────────────────

# Delayed neutron fraction for each of the 6 precursor groups
BETA_GROUPS = np.array([
    0.000215,   # group 1 — longest half-life  (T½ ≈ 55.7 s)
    0.001424,   # group 2
    0.001274,   # group 3
    0.002568,   # group 4
    0.000748,   # group 5
    0.000273,   # group 6 — shortest half-life (T½ ≈ 0.23 s)
])

# Decay constant λᵢ = ln(2) / T½  for each group [s⁻¹]
LAMBDA_GROUPS = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])

BETA_TOTAL = float(BETA_GROUPS.sum())   # total delayed neutron fraction ≈ 0.006502

LAMBDA_PROMPT = 2.0e-5   # Λ — prompt neutron generation time for thermal reactors [s]
                          # Note: this is the "average time from neutron birth to absorption"
                          # Thermal reactors: ~10⁻⁵ to 10⁻⁴ s
                          # Fast reactors:    ~10⁻⁷ s


# ─────────────────────────────────────────────────────────────────────────────
# Reactivity unit conversion
# ─────────────────────────────────────────────────────────────────────────────

def to_dimless(value: float, unit: str, beta: float) -> float:
    """Convert user-entered reactivity to dimensionless ρ."""
    if unit == "$":
        return value * beta           # 1 $ = β dimensionless
    return value * beta / 100.0       # 1 cent = β/100 dimensionless


def to_dollars(rho: float, beta: float) -> float:
    return rho / beta


def to_cents(rho: float, beta: float) -> float:
    return rho * 100.0 / beta


# ─────────────────────────────────────────────────────────────────────────────
# Reactivity profiles — ρ(t) as a callable for each insertion mode
# ─────────────────────────────────────────────────────────────────────────────

def step_rho(t: float, rho_val: float, t0: float) -> float:
    """Step insertion: ρ jumps from 0 to rho_val at t = t0."""
    return rho_val if t >= t0 else 0.0


def ramp_rho(t: float, rho_final: float, t_start: float, t_end: float) -> float:
    """Linear ramp from 0 to rho_final over [t_start, t_end], constant after."""
    if t < t_start:
        return 0.0
    if t > t_end:
        return rho_final
    return rho_final * (t - t_start) / (t_end - t_start)


def dynamic_rho(t: float, event_times: np.ndarray, cumulative_rhos: np.ndarray) -> float:
    """
    Piecewise-constant reactivity from a list of step events.
    Each event adds Δρ at the specified time; ρ(t) is the running sum.
    Uses binary search so it's fast even inside the ODE solver.
    """
    if len(event_times) == 0:
        return 0.0
    idx = int(np.searchsorted(event_times, t, side="right")) - 1
    return float(cumulative_rhos[idx]) if idx >= 0 else 0.0


def build_rho_func(args, beta: float):
    """Return a callable rho(t) based on parsed arguments."""
    mode = args.mode

    if mode == "step":
        rho_val = to_dimless(args.rho, args.unit, beta)
        return partial(step_rho, rho_val=rho_val, t0=args.t0)

    if mode == "ramp":
        rho_final = to_dimless(args.rho, args.unit, beta)
        return partial(ramp_rho, rho_final=rho_final,
                       t_start=args.ramp_start, t_end=args.ramp_end)

    # dynamic
    events = _parse_dynamic_events(args, beta)
    if not events:
        sys.exit("Dynamic mode requires at least one --event or --events-file.")
    events.sort(key=lambda e: e[0])
    ev_times = np.array([e[0] for e in events])
    ev_rhos  = np.cumsum([e[1] for e in events])
    return partial(dynamic_rho, event_times=ev_times, cumulative_rhos=ev_rhos)


def _parse_dynamic_events(args, beta: float) -> list[tuple[float, float]]:
    """Parse --event flags and optional --events-file into (time, rho) pairs."""
    events = []

    # Inline --event flags: format "TIME:VALUE:UNIT", e.g. "2.0:50:cents"
    for spec in (args.event or []):
        parts = spec.split(":")
        if len(parts) != 3:
            sys.exit(f"Bad --event format '{spec}'. Expected TIME:VALUE:UNIT "
                     f"e.g. '2.0:50:cents' or '5.0:-0.3:$'")
        t_ev   = float(parts[0])
        val    = float(parts[1])
        unit   = parts[2].strip()
        rho_ev = to_dimless(val, unit, beta)
        events.append((t_ev, rho_ev))

    # CSV events file: columns time, delta_rho, unit
    if args.events_file:
        with open(args.events_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_ev   = float(row["time"])
                val    = float(row["delta_rho"])
                unit   = row["unit"].strip()
                rho_ev = to_dimless(val, unit, beta)
                events.append((t_ev, rho_ev))

    return events


# ─────────────────────────────────────────────────────────────────────────────
# ODE right-hand side — the PKE system
# ─────────────────────────────────────────────────────────────────────────────

def pke_rhs(t, y, Lambda, beta_groups, lambda_groups, rho_func):
    """
    The coupled ODE system for point kinetics.

    State vector y has 7 elements:
      y[0]    = n(t)         neutron density (normalized)
      y[1..6] = C₁..C₆(t)   precursor group concentrations
    """
    n = max(y[0], 0.0)   # physical floor — n cannot go negative
    C = y[1:]            # 6-element precursor array
    beta = beta_groups.sum()
    rho  = rho_func(t)

    # ── Neutron density equation ──────────────────────────────────────────────
    # dn/dt = (ρ - β)/Λ · n  +  Σᵢ λᵢ Cᵢ
    #
    # Term 1:  (ρ - β)/Λ · n
    #   When ρ < β (subcritical/delayed-super): prompt neutrons alone cannot
    #   sustain the chain; this term is negative, damping n.
    #   When ρ ≥ β (prompt supercritical): prompt source exceeds losses → rapid rise.
    #
    # Term 2:  Σᵢ λᵢ Cᵢ
    #   Delayed neutrons released as precursors decay. This term is always
    #   positive and provides the "gentle" feedback that slows transients.
    dn = ((rho - beta) / Lambda) * n  +  np.dot(lambda_groups, C)

    # ── Precursor equations ───────────────────────────────────────────────────
    # dCᵢ/dt = (βᵢ/Λ) · n  −  λᵢ · Cᵢ
    #
    # Term 1:  (βᵢ/Λ) · n   — production proportional to fission rate
    # Term 2:  −λᵢ · Cᵢ     — exponential decay of each group
    dC = (beta_groups / Lambda) * n  -  lambda_groups * C

    return np.concatenate([[dn], dC])


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_pke(rho_func, t_end, Lambda, beta_groups, lambda_groups, n_points):
    """
    Solve the PKE system from t=0 to t=t_end.

    Returns
    -------
    t  : 1-D array of time points
    n  : neutron density n(t)/n₀
    C  : precursor concentrations, shape (6, len(t))
    C0 : initial precursor concentrations (used for normalization)
    """

    # ── Initial conditions (steady state at ρ = 0) ───────────────────────────
    # At steady state: dn/dt = 0 and dCᵢ/dt = 0
    # From dCᵢ/dt = 0:  (βᵢ/Λ)·n = λᵢ·Cᵢ  →  Cᵢ(0) = βᵢ / (λᵢ·Λ)  · n(0)
    n0  = 1.0                                         # normalized power
    C0  = beta_groups / (lambda_groups * Lambda)      # steady-state precursors
    y0  = np.concatenate([[n0], C0])                  # full 7-element state vector

    # ── Time grid ─────────────────────────────────────────────────────────────
    n_pts  = max(n_points, 2000)
    t_eval = np.linspace(0.0, t_end, n_pts)

    # ── Stiffness note ────────────────────────────────────────────────────────
    # The PKE system is STIFF.  "Stiff" means the ODE has components that vary
    # on wildly different time scales simultaneously:
    #   - Fastest: prompt neutron lifetime Λ ≈ 2×10⁻⁵ s  →  rate ~ 50,000 s⁻¹
    #   - Slowest: longest precursor group  λ₁ ≈ 0.012 s⁻¹ → time scale ~ 80 s
    #   - Stiffness ratio ≈ 50,000 / 0.012 ≈ 4,000,000
    #
    # Explicit methods (Euler, RK45) would need step sizes h < Λ = 2×10⁻⁵ s to
    # stay stable, requiring ~500,000 steps for a 10-second run — extremely slow.
    #
    # Implicit methods (Radau, BDF) are stable for any step size, so they can
    # take steps governed by accuracy rather than stability.
    #
    # Radau IIA (5th order, L-stable):
    #   - Designed specifically for stiff problems
    #   - L-stability means highly oscillatory stiff components are damped,
    #     not just bounded (unlike A-stable methods)
    #   - scipy's Radau adaptively controls step size using error estimates
    sol = solve_ivp(
        fun=partial(pke_rhs,
                    Lambda=Lambda,
                    beta_groups=beta_groups,
                    lambda_groups=lambda_groups,
                    rho_func=rho_func),
        t_span=(0.0, t_end),
        y0=y0,
        method="Radau",          # implicit, L-stable — required for stiff PKE
        t_eval=t_eval,
        rtol=1e-8,               # relative tolerance (controls accuracy)
        atol=1e-10,              # absolute tolerance
        max_step=min(0.005, t_end / 500),  # cap step to resolve reactivity events
    )

    if not sol.success:
        print(f"WARNING: solver did not converge — {sol.message}", file=sys.stderr)

    t = sol.t
    n = np.maximum(sol.y[0], 0.0)   # enforce physical floor
    C = sol.y[1:]                    # shape (6, len(t))
    return t, n, C, C0


# ─────────────────────────────────────────────────────────────────────────────
# Results analysis
# ─────────────────────────────────────────────────────────────────────────────

def reactor_period(t, n) -> float | None:
    """
    Estimate the asymptotic reactor period T = 1 / (d ln n / dt).
    Uses the last 20% of the time window where the transient has settled
    into an exponential growth/decay mode.
    Returns None if power is flat or decaying to zero.
    """
    tail = int(len(t) * 0.80)
    t_t, n_t = t[tail:], n[tail:]
    if n_t.min() <= 0 or n_t.max() / max(n_t.min(), 1e-30) < 1.005:
        return None
    try:
        slope, _ = np.polyfit(t_t, np.log(n_t), 1)
        return 1.0 / slope if abs(slope) > 1e-9 else None
    except Exception:
        return None


def prompt_jump(rho: float, beta: float) -> float:
    """
    Prompt-jump approximation for an instantaneous step insertion.

    After a step change in ρ, the neutron population makes an instantaneous
    "prompt jump" before the delayed neutrons respond.  For ρ < β:

      n_after / n_before = (β - ρ_before) / (β - ρ_after)

    For a step from ρ=0 to ρ:  n_pj / n₀ = β / (β - ρ)

    This is only valid for |ρ| << β.  For ρ → β, the approximation
    diverges, signalling that the prompt-jump assumption breaks down.
    """
    denom = beta - rho
    if denom <= 0:
        return float("inf")
    return beta / denom


def classify_state(rho_max_dollars: float) -> str:
    if rho_max_dollars >= 1.0:
        return "PROMPT SUPERCRITICAL  (ρ ≥ β)"
    if rho_max_dollars > 0:
        return "DELAYED SUPERCRITICAL  (0 < ρ < β)"
    if rho_max_dollars == 0.0:
        return "CRITICAL  (ρ = 0)"
    return "SUBCRITICAL  (ρ < 0)"


# ─────────────────────────────────────────────────────────────────────────────
# Output — text summary
# ─────────────────────────────────────────────────────────────────────────────

RED   = "\033[91m" if sys.stdout.isatty() else ""
RESET = "\033[0m"  if sys.stdout.isatty() else ""
BOLD  = "\033[1m"  if sys.stdout.isatty() else ""


def print_summary(args, t, n, C, C0, rho_vals, beta, lambda_groups):
    rho_max   = rho_vals.max()
    rho_final = rho_vals[-1]
    n_max_idx = int(np.argmax(n))

    period = reactor_period(t, n)
    pj     = prompt_jump(rho_final, beta) if args.mode == "step" else None

    state      = classify_state(to_dollars(rho_max, beta))
    is_prompt  = rho_max >= beta

    print(f"\n{BOLD}{'=' * 64}{RESET}")
    print(f"{BOLD}  Point Kinetics Equation (PKE) Simulation{RESET}")
    print(f"{'=' * 64}")
    print(f"  Mode        : {args.mode}")

    if args.mode in ("step", "ramp"):
        rho_in = to_dimless(args.rho, args.unit, beta)
        print(f"  Reactivity  : {to_cents(rho_in, beta):+.2f} ¢"
              f"  =  {to_dollars(rho_in, beta):+.4f} $"
              f"  (ρ = {rho_in:+.6f})")
    if args.mode == "step":
        print(f"  Insert at   : t₀ = {args.t0:.2f} s")
    if args.mode == "ramp":
        print(f"  Ramp        : {args.ramp_start:.2f} s → {args.ramp_end:.2f} s")

    print(f"  Parameters  : β = {beta:.6f},  "
          f"Λ = {args.Lambda:.2e} s,  t_end = {args.t_end:.1f} s")
    print(f"{'-' * 64}")

    if is_prompt:
        print(f"  {RED}{BOLD}Reactor state : {state}{RESET}")
        print(f"  {RED}WARNING: prompt supercritical — power rises on microsecond timescale{RESET}")
    else:
        print(f"  Reactor state : {state}")

    print()
    if period is not None:
        print(f"  Reactor period  T  = {period:.3f} s  "
              f"(asymptotic, from last 20% of window)")
    else:
        print(f"  Reactor period  T  = ∞  (power flat or decaying)")

    if pj is not None and not is_prompt:
        print(f"  Prompt-jump est    ≈ {pj:.4f} · n₀  (β / (β − ρ))")

    print(f"  Peak power    n_max = {n.max():.4f} · n₀  at t = {t[n_max_idx]:.3f} s")
    print(f"  Final power   n_fin = {n[-1]:.4f} · n₀")
    print()
    print(f"  Delayed precursor groups at t_end:")

    half_lives = np.log(2) / lambda_groups
    for i in range(6):
        ratio = C[i, -1] / C0[i] if C0[i] > 0 else float("nan")
        print(f"    Group {i+1}  "
              f"(λ = {lambda_groups[i]:6.4f} s⁻¹,  T½ = {half_lives[i]:5.2f} s)"
              f"   C/C₀ = {ratio:6.3f}")

    print(f"{'=' * 64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Output — CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(path: str, t, n, rho_vals, C, beta):
    header = ["t", "n", "rho_dim", "rho_dollars", "rho_cents",
              "C1", "C2", "C3", "C4", "C5", "C6"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, ti in enumerate(t):
            row = [
                f"{ti:.8f}",
                f"{n[i]:.8f}",
                f"{rho_vals[i]:.8f}",
                f"{to_dollars(rho_vals[i], beta):.8f}",
                f"{to_cents(rho_vals[i], beta):.8f}",
            ] + [f"{C[g, i]:.8f}" for g in range(6)]
            writer.writerow(row)
    print(f"  Saved CSV  → {path}  ({len(t)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Output — plots saved to PNG
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def save_plots(prefix: str, t, n, rho_vals, C, C0, beta, lambda_groups):
    # ── Power plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, n, color=COLORS[0], linewidth=1.8, label="n(t) / n₀")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
               label="n₀ (initial)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neutron Power  n(t) / n₀")
    ax.set_title("Neutron Power Transient")
    if n.max() > 50:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_p = f"{prefix}_power.png"
    fig.savefig(path_p, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {path_p}")

    # ── Reactivity plot ───────────────────────────────────────────────────────
    rho_d = rho_vals / beta    # dollars
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, rho_d, color=COLORS[3], linewidth=1.8, label="ρ(t)")
    ax.axhline(1.0, color="darkred", linestyle=":", linewidth=1.0,
               label="Prompt critical (1 $)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Reactivity  ρ  [$]")
    ax.set_title("Reactivity vs. Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.set_ylabel("Reactivity  ρ  [¢]")
    lim = ax.get_ylim()
    ax2.set_ylim(lim[0] * 100, lim[1] * 100)
    fig.tight_layout()
    path_r = f"{prefix}_rho.png"
    fig.savefig(path_r, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {path_r}")

    # ── Precursor plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    half_lives = np.log(2) / lambda_groups
    for i in range(6):
        norm = C[i] / C0[i] if C0[i] > 0 else C[i]
        ax.plot(t, norm, color=COLORS[i], linewidth=1.2,
                label=f"Group {i+1}  T½={half_lives[i]:.2f} s")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cᵢ(t) / Cᵢ(0)")
    ax.set_title("Delayed Neutron Precursor Concentrations (normalized)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_c = f"{prefix}_prec.png"
    fig.savefig(path_c, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {path_c}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pke_sim",
        description="Point Kinetics Equation simulation — nuclear reactor dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Step insertion of 50 cents:
    python pke/pke_sim.py --mode step --rho 50 --unit cents --t0 1 --t-end 30

  Ramp to 0.5$ over 4 seconds, save CSV and plots:
    python pke/pke_sim.py --mode ramp --rho 0.5 --unit $ --ramp-start 1 --ramp-end 5 \\
      --t-end 20 --out-csv ramp.csv --out-plot ramp

  Dynamic: insert +30¢ at t=2, remove -15¢ at t=8:
    python pke/pke_sim.py --mode dynamic \\
      --event 2.0:30:cents --event 8.0:-15:cents --t-end 25

  Custom prompt generation time (fast reactor: Λ = 1e-7 s):
    python pke/pke_sim.py --mode step --rho 50 --unit cents --Lambda 1e-7 --t-end 5
""",
    )

    # ── Reactor parameters ────────────────────────────────────────────────────
    rp = p.add_argument_group("Reactor parameters")
    rp.add_argument("--Lambda", type=float, default=LAMBDA_PROMPT, metavar="FLOAT",
                    help=f"Prompt neutron generation time Λ [s]  (default: {LAMBDA_PROMPT:.0e})")
    rp.add_argument("--beta", type=str, default=None, metavar="b1,b2,...,b6",
                    help="6 comma-separated beta group values  (default: Keepin U-235)")
    rp.add_argument("--t-end", type=float, default=30.0, metavar="FLOAT",
                    help="Simulation duration [s]  (default: 30)")
    rp.add_argument("--n-points", type=int, default=3000, metavar="INT",
                    help="Number of output time points  (default: 3000)")

    # ── Mode ─────────────────────────────────────────────────────────────────
    p.add_argument("--mode", choices=["step", "ramp", "dynamic"], default="step",
                   help="Reactivity insertion mode  (default: step)")

    # ── Step ─────────────────────────────────────────────────────────────────
    sm = p.add_argument_group("Step / Ramp mode parameters")
    sm.add_argument("--rho", type=float, default=50.0,
                    help="Reactivity value in --unit  (default: 50 cents)")
    sm.add_argument("--unit", choices=["$", "cents"], default="cents",
                    help="Reactivity unit: $ or cents  (default: cents)")
    sm.add_argument("--t0", type=float, default=0.0,
                    help="Step insertion time [s]  (default: 0)")
    sm.add_argument("--ramp-start", type=float, default=0.0,
                    help="Ramp start time [s]  (default: 0)")
    sm.add_argument("--ramp-end", type=float, default=5.0,
                    help="Ramp end time [s]  (default: 5)")

    # ── Dynamic ───────────────────────────────────────────────────────────────
    dm = p.add_argument_group("Dynamic mode parameters")
    dm.add_argument("--event", action="append", metavar="TIME:VALUE:UNIT",
                    help="Reactivity event  e.g. '2.0:+50:cents'  (repeatable)")
    dm.add_argument("--events-file", metavar="PATH",
                    help="CSV file with columns: time, delta_rho, unit")

    # ── Output ────────────────────────────────────────────────────────────────
    op = p.add_argument_group("Output")
    op.add_argument("--out-csv", metavar="PATH",
                    help="Save time-series to CSV file")
    op.add_argument("--out-plot", metavar="PREFIX",
                    help="Save plots as PREFIX_power.png, PREFIX_rho.png, PREFIX_prec.png")
    op.add_argument("--quiet", action="store_true",
                    help="Suppress text summary")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Beta groups — use custom values if provided, else Keepin defaults
    if args.beta:
        beta_groups = np.array([float(x) for x in args.beta.split(",")])
        if len(beta_groups) != 6:
            sys.exit("--beta requires exactly 6 comma-separated values.")
    else:
        beta_groups = BETA_GROUPS.copy()

    beta = float(beta_groups.sum())

    # Build ρ(t) callable for the chosen mode
    rho_func = build_rho_func(args, beta)

    # Solve the ODE system
    t, n, C, C0 = solve_pke(
        rho_func=rho_func,
        t_end=args.t_end,
        Lambda=args.Lambda,
        beta_groups=beta_groups,
        lambda_groups=LAMBDA_GROUPS,
        n_points=args.n_points,
    )

    # Reconstruct ρ(t) array for plotting / reporting
    rho_vals = np.array([rho_func(ti) for ti in t])

    # Text output
    if not args.quiet:
        print_summary(args, t, n, C, C0, rho_vals, beta, LAMBDA_GROUPS)

    # Optional CSV
    if args.out_csv:
        save_csv(args.out_csv, t, n, rho_vals, C, beta)

    # Optional plots
    if args.out_plot:
        save_plots(args.out_plot, t, n, rho_vals, C, C0, beta, LAMBDA_GROUPS)


if __name__ == "__main__":
    main()
