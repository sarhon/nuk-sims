# nuk-sims — Nuclear Reactor Point Kinetics Simulator

A from-scratch implementation of the **Point Kinetics Equations (PKE)** for nuclear reactor dynamics.  
Two interfaces: an interactive GUI (`pke_gui.py`) and a pure CLI simulation script (`pke_sim.py`).

---

## Table of Contents

1. [Physics Background](#physics-background)
2. [The ODE System](#the-ode-system)
3. [Reactivity Units](#reactivity-units)
4. [Reactor States](#reactor-states)
5. [Delayed Neutron Data](#delayed-neutron-data)
6. [Numerical Method](#numerical-method)
7. [GUI Script — `pke_gui.py`](#gui-script--pke_guipy)
8. [CLI Script — `pke_sim.py`](#cli-script--pke_simpy)
9. [Output Formats](#output-formats)
10. [Example Scenarios](#example-scenarios)
11. [Installation](#installation)

---

## Physics Background

In a nuclear reactor, the neutron population at any moment is the result of a competition between
neutron production (fission) and neutron losses (absorption, leakage).  **Point kinetics** is a
simplification that treats the entire reactor core as a single spatial "point," collapsing the full
neutron transport problem into a coupled set of ordinary differential equations (ODEs).

Two neutron populations drive the dynamics:

| Type | Source | Timescale |
|------|--------|-----------|
| **Prompt neutrons** | Emitted directly from fission, within ~10⁻¹⁴ s | Λ ≈ 2×10⁻⁵ s (thermal) |
| **Delayed neutrons** | Emitted from fission-product precursor decay | 0.2 s – 80 s (6 groups) |

Although delayed neutrons are only ~0.65% of all fission neutrons, they are critical: without them,
a reactor controlled to *k* slightly above 1 would respond on a microsecond timescale, making manual
or mechanical control impossible.

---

## The ODE System

The PKE model consists of **7 coupled ODEs** — one for neutron density and one per precursor group.

### Neutron Density

$$\frac{dn}{dt} = \frac{\rho(t) - \beta}{\Lambda}\, n(t) \;+\; \sum_{i=1}^{6} \lambda_i C_i(t)$$

| Term | Meaning |
|------|---------|
| $\frac{\rho - \beta}{\Lambda}\, n$ | Net prompt source. Negative when $\rho < \beta$ (delayed supercritical), positive when $\rho \ge \beta$ (prompt supercritical). |
| $\sum \lambda_i C_i$ | Delayed neutron source — precursors decay and release neutrons. Always positive; provides the slow, controllable feedback. |

### Precursor Concentrations (i = 1 … 6)

$$\frac{dC_i}{dt} = \frac{\beta_i}{\Lambda}\, n(t) \;-\; \lambda_i\, C_i(t)$$

| Term | Meaning |
|------|---------|
| $\frac{\beta_i}{\Lambda}\, n$ | Production — proportional to fission rate |
| $-\lambda_i C_i$ | Exponential decay of precursor group *i* |

### Variables

| Symbol | Description | Typical value |
|--------|-------------|---------------|
| $n(t)$ | Neutron density, normalized so $n(0) = 1$ | — |
| $C_i(t)$ | Precursor group *i* concentration | — |
| $\rho(t)$ | Reactivity (dimensionless) | — |
| $\beta$ | Total delayed neutron fraction $= \sum \beta_i$ | 0.006502 (U-235) |
| $\beta_i$ | Fraction for precursor group *i* | see table below |
| $\Lambda$ | Prompt neutron generation time | 2×10⁻⁵ s (thermal) |
| $\lambda_i$ | Decay constant for group *i* | see table below |

### Steady-State Initial Conditions

At $t = 0$ the reactor is at steady state ($\rho = 0$, power constant).  
Setting $dn/dt = 0$ and $dC_i/dt = 0$:

$$n(0) = 1, \qquad C_i(0) = \frac{\beta_i}{\lambda_i \,\Lambda}$$

---

## Reactivity Units

Reactivity $\rho$ is dimensionless, but reactor engineers use two scaled units for convenience:

| Unit | Definition | Physical meaning |
|------|------------|-----------------|
| **dollars** ($) | rho_$ = rho / beta | rho = 1 $ is exactly prompt critical |
| **cents** (¢) | rho_cents = 100 * rho_$ | 1/100th of a dollar |

**Conversion example** with $\beta = 0.006502$:

```
50 ¢  →  0.50 $  →  ρ = 0.50 × 0.006502 = 0.003251  (dimensionless)
```

Both scripts accept `$` or `cents` as the `--unit` argument and display results in all three forms.

---

## Reactor States

| Condition | Name | Behavior |
|-----------|------|----------|
| $\rho < 0$ (rho_$ < 0) | **Subcritical** | Power decreases exponentially |
| $\rho = 0$ (rho_$ = 0) | **Critical** | Steady state, constant power |
| $0 < \rho < \beta$ (0 < rho_$ < 1) | **Delayed supercritical** | Slow power rise governed by precursor decay (seconds to minutes) |
| $\rho \ge \beta$ (rho_$ >= 1) | **Prompt supercritical** | Prompt neutrons alone sustain reaction; power rises on microsecond timescale |

The delayed supercritical regime is the normal operating range for power changes.  
Prompt supercritical is the condition during reactor accidents (e.g. Chernobyl Unit 4).

---

## Delayed Neutron Data

Default data: **Keepin 6-group model for U-235 thermal fission (1965)**.

| Group | $\beta_i$ | $\lambda_i$ (s⁻¹) | Half-life $T_{1/2}$ (s) |
|-------|-----------|-------------------|------------------------|
| 1 | 0.000215 | 0.0124 | 55.90 |
| 2 | 0.001424 | 0.0305 | 22.73 |
| 3 | 0.001274 | 0.1110 | 6.24 |
| 4 | 0.002568 | 0.3010 | 2.30 |
| 5 | 0.000748 | 1.1400 | 0.61 |
| 6 | 0.000273 | 3.0100 | 0.23 |
| **Total** | **0.006502** | — | — |

Custom $\beta_i$ values can be provided via the GUI (edit β groups panel) or the CLI (`--beta` flag).

---

## Numerical Method

### Why the PKE is stiff

"Stiff" ODEs have components that evolve on vastly different timescales simultaneously.  
For the PKE:

| Component | Characteristic rate |
|-----------|-------------------|
| Prompt neutron dynamics | $\sim 1/\Lambda \approx 50{,}000\ \text{s}^{-1}$ |
| Slowest precursor group | $\lambda_1 \approx 0.012\ \text{s}^{-1}$ |
| **Stiffness ratio** | **≈ 4,000,000** |

An explicit solver (Euler, RK45) would need step sizes $h < \Lambda = 2\times10^{-5}$ s to stay
numerically stable — requiring ~500,000 steps per 10 seconds of simulation, and still being slow
and inaccurate near the step discontinuity.

### Radau IIA (implicit Runge-Kutta)

Both scripts use `scipy.integrate.solve_ivp` with `method='Radau'`:

- **L-stable** — highly oscillatory stiff components are damped, not just bounded
- **5th order** — high accuracy per step
- **Adaptive step control** — step size governed by accuracy, not stability
- Tolerances: `rtol=1e-8`, `atol=1e-10`
- `max_step` is capped to prevent striding across reactivity discontinuities

### Prompt-Jump Approximation

For a step insertion, after the initial fast transient the neutron density makes an instantaneous
"prompt jump" before the precursors have time to respond:

$$\frac{n^+}{n^-} = \frac{\beta - \rho_{\rm before}}{\beta - \rho_{\rm after}}$$

For a step from $\rho = 0$ to $\rho$:

$$\frac{n_{\rm pj}}{n_0} = \frac{\beta}{\beta - \rho}$$

This analytical estimate is printed alongside the numerical result as a sanity check.  
It is only accurate for $|\rho| \ll \beta$.

---

## GUI Script — `pke_gui.py`

Interactive Tkinter application.  Adjust parameters and see the transient update in real time.

### Run

```bash
uv run python pke/pke_gui.py
```

### Layout

```
┌─── Sidebar (320 px) ───┬─── Plot Area ─────────────────────────┐
│                         │                                        │
│  Reactor Parameters     │  [1] Neutron Power  n(t)/n₀           │
│    Λ (gen. time)        │      log scale toggle available        │
│    Simulation duration  │                                        │
│    Edit β groups        │  [2] Reactivity ρ(t)                   │
│    (collapsible)        │      left axis: $,  right axis: ¢      │
│                         │      dashed line at 1 $ (prompt crit.) │
│  Insertion Mode         │                                        │
│    ● Step               │  [3] Precursor Concentrations          │
│    ○ Ramp               │      all 6 groups, normalized C/C₀     │
│    ○ Dynamic            │                                        │
│                         │  NavigationToolbar (pan / zoom)        │
│  [Mode Parameters]      │                                        │
│                         │  Status bar:                           │
│  ☐ Log scale            │    Period | Max power | Final power     │
│  ▶ Run Simulation       │    ⚠ PROMPT SUPERCRITICAL (if ρ ≥ β)  │
└─────────────────────────┴────────────────────────────────────────┘
```

### Insertion Modes

**Step** — instantaneous reactivity change at time $t_0$  
Enter: reactivity value, unit ($ or ¢), insertion time

**Ramp** — linear increase from 0 to $\rho_{\rm final}$ over $[t_{\rm start},\, t_{\rm end}]$, constant after  
Enter: final reactivity, unit, start time, end time

**Dynamic** — arbitrary sequence of step events  
Enter events one at a time (time + Δρ + unit); each adds to a running cumulative table.  
Delete individual events with the × button.

---

## CLI Script — `pke_sim.py`

Pure simulation — no window, no interaction.  Provide inputs as flags, get text output and
optional saved files.  Designed to be readable as a teaching document.

### Run

```bash
uv run python pke/pke_sim.py [OPTIONS]
```

### All Options

#### Reactor Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--Lambda FLOAT` | `2e-5` | Prompt neutron generation time Λ [s] |
| `--beta b1,...,b6` | Keepin U-235 | 6 comma-separated $\beta_i$ values |
| `--t-end FLOAT` | `30` | Simulation duration [s] |
| `--n-points INT` | `3000` | Number of output time points |

#### Insertion Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {step,ramp,dynamic}` | `step` | Reactivity insertion profile |

#### Step Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--rho FLOAT` | `50` | Reactivity magnitude |
| `--unit {$,cents}` | `cents` | Unit for `--rho` |
| `--t0 FLOAT` | `0.0` | Insertion time [s] |

#### Ramp Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--rho FLOAT` | — | Final reactivity value |
| `--unit {$,cents}` | `cents` | Unit for `--rho` |
| `--ramp-start FLOAT` | `0.0` | Start of ramp [s] |
| `--ramp-end FLOAT` | `5.0` | End of ramp [s] |

#### Dynamic Mode

Use `--event` flags (repeatable) **or** a CSV file — or both:

| Flag | Format | Description |
|------|--------|-------------|
| `--event` | `TIME:VALUE:UNIT` | Single reactivity event, e.g. `2.0:+50:cents` |
| `--events-file PATH` | CSV | File with columns `time, delta_rho, unit` |

Events are cumulative: the second `--event` adds to whatever ρ is already present.

#### Output

| Flag | Description |
|------|-------------|
| `--out-csv PATH` | Save full time-series to CSV |
| `--out-plot PREFIX` | Save three PNGs: `PREFIX_power.png`, `PREFIX_rho.png`, `PREFIX_prec.png` |
| `--quiet` | Suppress stdout text summary |

---

## Output Formats

### Text Summary (stdout)

```
================================================================
  Point Kinetics Equation (PKE) Simulation
================================================================
  Mode        : step
  Reactivity  : +50.00 ¢  =  +0.5000 $  (ρ = +0.003251)
  Insert at   : t₀ = 1.00 s
  Parameters  : β = 0.006502,  Λ = 2.00e-05 s,  t_end = 30.0 s
----------------------------------------------------------------
  Reactor state : DELAYED SUPERCRITICAL  (0 < ρ < β)

  Reactor period  T  = 5.731 s  (asymptotic, last 20% of window)
  Prompt-jump est    ≈ 2.0000 · n₀  (β / (β − ρ))
  Peak power    n_max = 426.9481 · n₀  at t = 30.000 s
  Final power   n_fin = 426.9481 · n₀

  Delayed precursor groups at t_end:
    Group 1  (λ = 0.0124 s⁻¹,  T½ = 55.90 s)   C/C₀ = 28.852
    ...
================================================================
```

Prompt supercritical cases print a red warning line.

### CSV Columns

| Column | Description |
|--------|-------------|
| `t` | Time [s] |
| `n` | Neutron power, normalized ($n/n_0$) |
| `rho_dim` | Reactivity (dimensionless) |
| `rho_dollars` | Reactivity [$] |
| `rho_cents` | Reactivity [¢] |
| `C1` – `C6` | Precursor group concentrations |

### Plot Files

| File | Contents |
|------|----------|
| `PREFIX_power.png` | $n(t)/n_0$ vs time; log scale auto-applied if peak > 50 |
| `PREFIX_rho.png` | $\rho(t)$ in $ (left axis) and ¢ (right axis); prompt critical line at 1 $ |
| `PREFIX_prec.png` | All 6 precursor groups, normalized $C_i / C_i(0)$ |

---

## Example Scenarios

### 1. Small step insertion (delayed supercritical)

```bash
uv run python pke/pke_sim.py --mode step --rho 50 --unit cents --t0 1 --t-end 30
```

**Expected:** Period ≈ 5.7 s, power rises to ~427 n₀ by t = 30 s.  
Slow rise driven entirely by precursor buildup — classic delayed supercritical transient.

---

### 2. Negative reactivity (subcritical insertion)

```bash
uv run python pke/pke_sim.py --mode step --rho -50 --unit cents --t0 1 --t-end 60
```

**Expected:** Power drops rapidly then levels off at ~0.25 n₀.  
The prompt drop is immediate; precursors continue providing a floor.

---

### 3. Ramp insertion

```bash
uv run python pke/pke_sim.py --mode ramp --rho 0.5 --unit '$' \
  --ramp-start 1 --ramp-end 5 --t-end 20
```

**Expected:** Gradual power rise during the ramp, then exponential growth after it holds.  
Compare with the step case to see how the ramp delays the transient onset.

---

### 4. Dynamic — insert then partially withdraw

```bash
uv run python pke/pke_sim.py --mode dynamic \
  --event 2.0:+30:cents \
  --event 8.0:-15:cents \
  --t-end 25
```

**Expected:** Power rises after t = 2 s (net +30 ¢), then slows after t = 8 s (net +15 ¢).  
Composite transient shows the asymptotic period shifting as ρ changes.

---

### 5. Prompt supercritical

```bash
uv run python pke/pke_sim.py --mode step --rho 1.05 --unit '$' --t0 0 --t-end 0.001
```

**Expected:** Red warning, power rises ~34% in just 1 ms.  
Note the very short `--t-end` — this transient is over in milliseconds, not seconds.

---

### 6. Fast reactor (small Λ)

```bash
uv run python pke/pke_sim.py --mode step --rho 50 --unit cents --Lambda 1e-7 --t-end 30
```

**Expected:** Similar asymptotic period to the thermal case (period is dominated by
precursor half-lives, not Λ) but the prompt jump is much sharper.

---

### 7. Save everything

```bash
uv run python pke/pke_sim.py --mode step --rho 50 --unit cents --t0 1 --t-end 30 \
  --out-csv results/step50c.csv \
  --out-plot results/step50c
```

Produces `step50c.csv`, `step50c_power.png`, `step50c_rho.png`, `step50c_prec.png`.

---

## Installation

Requires Python 3.14+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo>
cd nuk-sims
uv sync          # installs numpy, scipy, matplotlib into .venv
```

Run the GUI:
```bash
uv run python pke/pke_gui.py
```

Run the CLI:
```bash
uv run python pke/pke_sim.py --help
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array math, linear algebra |
| `scipy` | `solve_ivp` with Radau solver |
| `matplotlib` | Plots (GUI canvas + PNG file output) |
| `scipy-stubs` | IDE type hints for scipy |
