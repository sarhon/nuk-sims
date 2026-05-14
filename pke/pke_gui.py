"""
Point Kinetics Equation (PKE) Simulator
Nuclear Reactor Dynamics — from first principles

ODE system (7 equations: 1 neutron density + 6 precursor groups):

  dn/dt   = [(ρ(t) - β) / Λ] · n(t) + Σᵢ λᵢ · Cᵢ(t)
  dCᵢ/dt  = (βᵢ / Λ) · n(t) − λᵢ · Cᵢ(t)    i = 1..6

Steady-state ICs (ρ = 0):  n₀ = 1,  Cᵢ₀ = βᵢ / (λᵢ · Λ)
Reactivity units:  ρ_$ = ρ / β,  ρ_¢ = 100 · ρ_$
Prompt supercritical threshold: ρ_$ ≥ 1.0
"""

from __future__ import annotations

import itertools
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp

matplotlib.use("TkAgg")

# ---------------------------------------------------------------------------
# Physics constants — Keepin 6-group U-235 data
# ---------------------------------------------------------------------------

BETA_GROUPS = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
LAMBDA_GROUPS = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])  # s⁻¹
BETA_TOTAL = float(BETA_GROUPS.sum())  # 0.006502

DEFAULT_LAMBDA_PROMPT = 2.0e-5  # Λ, prompt neutron generation time [s]
DEFAULT_T_END = 10.0            # simulation duration [s]
N_GROUPS = 6


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def to_dimensionless(value: float, unit: str, beta: float) -> float:
    """Convert user-entered reactivity to dimensionless ρ."""
    if unit == "$":
        return value * beta
    return value * beta / 100.0  # cents


def from_dimensionless(rho: float, unit: str, beta: float) -> float:
    """Convert dimensionless ρ to display unit."""
    if unit == "$":
        return rho / beta
    return rho * 100.0 / beta  # cents


# ---------------------------------------------------------------------------
# Reactivity profile — encodes ρ(t) for all three insertion modes
# ---------------------------------------------------------------------------

class ReactivityProfile:
    """Pure data class encoding ρ(t). No GUI or solver dependency."""

    def __init__(self, mode: str, **kw):
        self._mode = mode
        self._kw = kw
        if mode == "dynamic":
            events = sorted(kw["events"], key=lambda e: e[0])
            if events:
                self._ev_times = np.array([e[0] for e in events])
                self._ev_rhos = np.cumsum([e[1] for e in events])
            else:
                self._ev_times = np.array([])
                self._ev_rhos = np.array([])

    @classmethod
    def step(cls, rho_insert: float, t0: float = 0.0) -> ReactivityProfile:
        return cls("step", rho_insert=rho_insert, t0=t0)

    @classmethod
    def ramp(cls, rho_final: float, t_start: float, t_end: float) -> ReactivityProfile:
        return cls("ramp", rho_final=rho_final, t_start=t_start, t_end=t_end)

    @classmethod
    def dynamic(cls, events: list[tuple[float, float]]) -> ReactivityProfile:
        """events = list of (time, delta_rho) pairs."""
        return cls("dynamic", events=events)

    def rho_at(self, t: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        result = np.zeros_like(t_arr)

        if self._mode == "step":
            t0 = self._kw["t0"]
            rho = self._kw["rho_insert"]
            result = np.where(t_arr >= t0, rho, 0.0)

        elif self._mode == "ramp":
            t_s = self._kw["t_start"]
            t_e = self._kw["t_end"]
            rho_f = self._kw["rho_final"]
            duration = t_e - t_s
            if duration <= 0:
                result = np.where(t_arr >= t_s, rho_f, 0.0)
            else:
                in_ramp = (t_arr >= t_s) & (t_arr <= t_e)
                past_ramp = t_arr > t_e
                result = np.where(in_ramp, rho_f * (t_arr - t_s) / duration, result)
                result = np.where(past_ramp, rho_f, result)

        elif self._mode == "dynamic":
            if len(self._ev_times) == 0:
                pass  # remains zero
            else:
                idx = np.searchsorted(self._ev_times, t_arr, side="right") - 1
                mask = idx >= 0
                result[mask] = self._ev_rhos[idx[mask]]

        return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class ResultsBundle:
    t: np.ndarray
    n: np.ndarray
    C: np.ndarray          # shape (6, N)
    rho: np.ndarray        # dimensionless
    beta_total: float
    Lambda: float
    success: bool
    message: str
    C0: np.ndarray = field(default_factory=lambda: np.ones(N_GROUPS))

    @property
    def rho_dollars(self) -> np.ndarray:
        return self.rho / self.beta_total

    @property
    def rho_cents(self) -> np.ndarray:
        return self.rho_dollars * 100.0

    @property
    def max_power(self) -> float:
        return float(self.n.max())

    @property
    def final_power(self) -> float:
        return float(self.n[-1])

    @property
    def reactor_period(self) -> float | None:
        tail = int(len(self.t) * 0.80)
        t_tail = self.t[tail:]
        n_tail = self.n[tail:]
        if n_tail.min() <= 0 or n_tail.max() / max(n_tail.min(), 1e-30) < 1.005:
            return None
        try:
            slope, _ = np.polyfit(t_tail, np.log(n_tail), 1)
        except Exception:
            return None
        if abs(slope) < 1e-9:
            return None
        return 1.0 / slope


class PKESolver:
    def __init__(
        self,
        profile: ReactivityProfile,
        Lambda: float,
        beta_groups: np.ndarray,
        lambda_groups: np.ndarray,
        t_end: float,
    ):
        self.profile = profile
        self.Lambda = Lambda
        self.beta_groups = beta_groups.copy()
        self.lambda_groups = lambda_groups.copy()
        self.beta = float(beta_groups.sum())
        self.t_end = t_end

    def _initial_conditions(self) -> np.ndarray:
        n0 = 1.0
        C0 = self.beta_groups / (self.lambda_groups * self.Lambda)
        return np.concatenate([[n0], C0])

    def _rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        n = max(y[0], 0.0)  # physical floor
        C = y[1:]
        rho = self.profile.rho_at(t)
        dn = ((rho - self.beta) / self.Lambda) * n + float(np.dot(self.lambda_groups, C))
        dC = (self.beta_groups / self.Lambda) * n - self.lambda_groups * C
        return np.concatenate([[dn], dC])

    def solve(self) -> ResultsBundle:
        y0 = self._initial_conditions()
        C0 = y0[1:].copy()
        n_pts = max(3000, int(self.t_end * 500))
        t_eval = np.linspace(0.0, self.t_end, n_pts)
        max_step = min(0.005, self.t_end / 500)

        sol = solve_ivp(
            fun=self._rhs,
            t_span=(0.0, self.t_end),
            y0=y0,
            method="Radau",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            max_step=max_step,
        )

        t = sol.t
        n = np.maximum(sol.y[0], 0.0)
        C = sol.y[1:]
        rho = np.array([self.profile.rho_at(ti) for ti in t])

        return ResultsBundle(
            t=t, n=n, C=C, rho=rho,
            beta_total=self.beta,
            Lambda=self.Lambda,
            success=sol.success,
            message=sol.message,
            C0=C0,
        )


# ---------------------------------------------------------------------------
# GUI — Mode frames
# ---------------------------------------------------------------------------

class _ModeFrame(ttk.LabelFrame):
    """Base class for mode-specific parameter frames."""

    def build_profile(self, beta_total: float) -> ReactivityProfile:
        raise NotImplementedError


class StepFrame(_ModeFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Step Insertion Parameters", padding=6)
        self._rho_var = tk.StringVar(value="50")
        self._unit_var = tk.StringVar(value="¢")
        self._t0_var = tk.StringVar(value="1.0")
        self._conv_var = tk.StringVar(value="")
        self._build()

    def _build(self):
        ttk.Label(self, text="Reactivity:").grid(row=0, column=0, sticky="w", pady=2)
        e = ttk.Entry(self, textvariable=self._rho_var, width=10)
        e.grid(row=0, column=1, padx=2)
        e.bind("<FocusOut>", self._update_conv)
        ttk.OptionMenu(self, self._unit_var, "¢", "$", "¢").grid(row=0, column=2, padx=2)
        ttk.Label(self, textvariable=self._conv_var, foreground="gray").grid(
            row=1, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(self, text="Insert at t:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(self, textvariable=self._t0_var, width=10).grid(row=2, column=1, padx=2)
        ttk.Label(self, text="s").grid(row=2, column=2, sticky="w")

    def _update_conv(self, _=None):
        try:
            v = float(self._rho_var.get())
            u = self._unit_var.get()
            rho = to_dimensionless(v, u, BETA_TOTAL)
            self._conv_var.set(
                f"= {rho/BETA_TOTAL:.4f} $  = {rho/BETA_TOTAL*100:.2f} ¢  (ρ = {rho:.5f})"
            )
        except ValueError:
            self._conv_var.set("")

    def build_profile(self, beta_total: float) -> ReactivityProfile:
        rho = to_dimensionless(float(self._rho_var.get()), self._unit_var.get(), beta_total)
        t0 = float(self._t0_var.get())
        return ReactivityProfile.step(rho, t0)


class RampFrame(_ModeFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Ramp Insertion Parameters", padding=6)
        self._rho_var = tk.StringVar(value="50")
        self._unit_var = tk.StringVar(value="¢")
        self._t_start_var = tk.StringVar(value="1.0")
        self._t_end_var = tk.StringVar(value="5.0")
        self._conv_var = tk.StringVar(value="")
        self._build()

    def _build(self):
        ttk.Label(self, text="Final ρ:").grid(row=0, column=0, sticky="w", pady=2)
        e = ttk.Entry(self, textvariable=self._rho_var, width=10)
        e.grid(row=0, column=1, padx=2)
        e.bind("<FocusOut>", self._update_conv)
        ttk.OptionMenu(self, self._unit_var, "¢", "$", "¢").grid(row=0, column=2, padx=2)
        ttk.Label(self, textvariable=self._conv_var, foreground="gray").grid(
            row=1, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(self, text="Ramp start:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(self, textvariable=self._t_start_var, width=10).grid(row=2, column=1, padx=2)
        ttk.Label(self, text="s").grid(row=2, column=2, sticky="w")
        ttk.Label(self, text="Ramp end:").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(self, textvariable=self._t_end_var, width=10).grid(row=3, column=1, padx=2)
        ttk.Label(self, text="s").grid(row=3, column=2, sticky="w")

    def _update_conv(self, _=None):
        try:
            v = float(self._rho_var.get())
            u = self._unit_var.get()
            rho = to_dimensionless(v, u, BETA_TOTAL)
            self._conv_var.set(
                f"= {rho/BETA_TOTAL:.4f} $  = {rho/BETA_TOTAL*100:.2f} ¢  (ρ = {rho:.5f})"
            )
        except ValueError:
            self._conv_var.set("")

    def build_profile(self, beta_total: float) -> ReactivityProfile:
        rho = to_dimensionless(float(self._rho_var.get()), self._unit_var.get(), beta_total)
        return ReactivityProfile.ramp(rho, float(self._t_start_var.get()), float(self._t_end_var.get()))


class DynamicFrame(_ModeFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Dynamic Reactivity Events", padding=6)
        self._events: list[dict] = []  # {"time": float, "delta_rho": float (dim'less)}
        self._add_time_var = tk.StringVar(value="2.0")
        self._add_rho_var = tk.StringVar(value="30")
        self._add_unit_var = tk.StringVar(value="¢")
        self._build()

    def _build(self):
        add_frame = ttk.LabelFrame(self, text="Add Event", padding=4)
        add_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(add_frame, text="Time:").grid(row=0, column=0, sticky="w")
        ttk.Entry(add_frame, textvariable=self._add_time_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(add_frame, text="s").grid(row=0, column=2, sticky="w")
        ttk.Label(add_frame, text="Δρ:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(add_frame, textvariable=self._add_rho_var, width=8).grid(row=1, column=1, padx=2)
        ttk.OptionMenu(add_frame, self._add_unit_var, "¢", "$", "¢").grid(row=1, column=2, padx=2)
        ttk.Button(add_frame, text="+ Add Event", command=self._add_event).grid(
            row=2, column=0, columnspan=3, pady=4
        )

        self._table_frame = ttk.Frame(self)
        self._table_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=4)

        ttk.Button(self, text="Clear All", command=self._clear_all).grid(
            row=2, column=0, columnspan=2, pady=2
        )
        self._rebuild_table()

    def _add_event(self):
        try:
            t = float(self._add_time_var.get())
            drho_raw = float(self._add_rho_var.get())
            drho = to_dimensionless(drho_raw, self._add_unit_var.get(), BETA_TOTAL)
        except ValueError:
            messagebox.showerror("Input Error", "Enter valid numbers for time and Δρ.")
            return
        self._events.append({"time": t, "delta_rho": drho})
        self._events.sort(key=lambda e: e["time"])
        self._rebuild_table()

    def _remove_event(self, idx: int):
        self._events.pop(idx)
        self._rebuild_table()

    def _clear_all(self):
        self._events.clear()
        self._rebuild_table()

    def _rebuild_table(self):
        for w in self._table_frame.winfo_children():
            w.destroy()

        headers = ["Time [s]", "Δρ [$]", "Δρ [¢]", "ρ_cum [$]", ""]
        for col, h in enumerate(headers):
            ttk.Label(self._table_frame, text=h, font=("TkDefaultFont", 8, "bold")).grid(
                row=0, column=col, padx=3, sticky="w"
            )

        cum_rho = list(itertools.accumulate(e["delta_rho"] for e in self._events))
        for row_i, (evt, cr) in enumerate(zip(self._events, cum_rho), start=1):
            drho_d = evt["delta_rho"] / BETA_TOTAL
            sign = "+" if drho_d >= 0 else ""
            vals = [
                f"{evt['time']:.2f}",
                f"{sign}{drho_d:.4f}",
                f"{sign}{drho_d*100:.2f}",
                f"{cr/BETA_TOTAL:.4f}",
            ]
            for col, v in enumerate(vals):
                ttk.Label(self._table_frame, text=v, font=("TkFixedFont", 8)).grid(
                    row=row_i, column=col, padx=3, sticky="w"
                )
            ttk.Button(
                self._table_frame, text="×", width=2,
                command=lambda i=row_i - 1: self._remove_event(i),
            ).grid(row=row_i, column=4, padx=2)

    def build_profile(self, beta_total: float) -> ReactivityProfile:
        events = [(e["time"], e["delta_rho"]) for e in self._events]
        return ReactivityProfile.dynamic(events)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class PKEApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PKE Nuclear Reactor Simulator")
        self.geometry("1280x860")
        self.minsize(960, 640)

        # --- Reactor parameter variables ---
        self._lambda_var = tk.StringVar(value=str(DEFAULT_LAMBDA_PROMPT))
        self._t_end_var = tk.StringVar(value=str(DEFAULT_T_END))
        self._log_scale_var = tk.BooleanVar(value=False)
        self._mode_var = tk.StringVar(value="step")
        self._beta_entry_vars = [tk.StringVar(value=str(v)) for v in BETA_GROUPS]
        self._beta_total_label_var = tk.StringVar(value=f"β total = {BETA_TOTAL:.6f}")
        self._ax_rho2 = None

        self._build_layout()
        self._run_simulation()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self, width=330, relief="flat")
        sidebar.grid(row=0, column=0, sticky="ns", padx=6, pady=6)
        sidebar.grid_propagate(False)

        plot_frame = ttk.Frame(self)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=6)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self._build_sidebar(sidebar)
        self._build_plot_area(plot_frame)

    def _build_sidebar(self, parent):
        # Scrollable canvas for the sidebar
        canvas = tk.Canvas(parent, width=318, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        canvas_win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_resize(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_win, width=event.width)

        inner.bind("<Configure>", _on_resize)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_win, width=e.width))

        # Mouse-wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._fill_sidebar(inner)

    def _fill_sidebar(self, parent):
        pad = {"padx": 6, "pady": 3}

        # --- Reactor Parameters ---
        rp = ttk.LabelFrame(parent, text="Reactor Parameters", padding=6)
        rp.pack(fill="x", **pad)

        ttk.Label(rp, text="Prompt gen. time Λ:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(rp, textvariable=self._lambda_var, width=12).grid(row=0, column=1, padx=4)
        ttk.Label(rp, text="s").grid(row=0, column=2, sticky="w")

        ttk.Label(rp, text="Sim. duration:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(rp, textvariable=self._t_end_var, width=12).grid(row=1, column=1, padx=4)
        ttk.Label(rp, text="s").grid(row=1, column=2, sticky="w")

        # Beta groups (collapsible via a toggle)
        self._beta_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            rp, text="Edit β groups", variable=self._beta_visible,
            command=lambda: self._toggle_beta_frame(beta_inner)
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)

        beta_inner = ttk.Frame(rp)
        # (hidden initially; _toggle_beta_frame manages grid)
        group_labels = ["β₁", "β₂", "β₃", "β₄", "β₅", "β₆"]
        for i, (lbl, var) in enumerate(zip(group_labels, self._beta_entry_vars)):
            ttk.Label(beta_inner, text=f"{lbl}:").grid(row=i, column=0, sticky="w")
            e = ttk.Entry(beta_inner, textvariable=var, width=10)
            e.grid(row=i, column=1, padx=4, pady=1)
            e.bind("<FocusOut>", self._refresh_beta_total)
            lval = LAMBDA_GROUPS[i]
            ttk.Label(beta_inner, text=f"λ={lval} s⁻¹", foreground="gray").grid(
                row=i, column=2, sticky="w"
            )
        ttk.Button(
            beta_inner, text="Reset to Keepin defaults",
            command=self._reset_beta_defaults
        ).grid(row=N_GROUPS, column=0, columnspan=3, pady=4)

        ttk.Label(rp, textvariable=self._beta_total_label_var, foreground="navy").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=2
        )

        # --- Insertion Mode ---
        im = ttk.LabelFrame(parent, text="Reactivity Insertion Mode", padding=6)
        im.pack(fill="x", **pad)

        modes = [("Step (instantaneous)", "step"),
                 ("Ramp (linear)", "ramp"),
                 ("Dynamic (events)", "dynamic")]
        for lbl, val in modes:
            ttk.Radiobutton(
                im, text=lbl, variable=self._mode_var, value=val,
                command=self._switch_mode
            ).pack(anchor="w")

        # --- Mode frames ---
        self._mode_container = ttk.Frame(parent)
        self._mode_container.pack(fill="x", **pad)

        self._step_frame = StepFrame(self._mode_container)
        self._ramp_frame = RampFrame(self._mode_container)
        self._dynamic_frame = DynamicFrame(self._mode_container)

        self._step_frame.pack(fill="x")  # visible by default

        # --- Options + Run ---
        opts = ttk.Frame(parent)
        opts.pack(fill="x", **pad)
        ttk.Checkbutton(
            opts, text="Log scale (power plot)", variable=self._log_scale_var,
            command=self._toggle_log_scale
        ).pack(anchor="w", pady=2)

        ttk.Button(
            parent, text="▶  Run Simulation", command=self._run_simulation,
            style="Accent.TButton"
        ).pack(fill="x", padx=6, pady=8, ipady=6)

        # --- Status ---
        status_frame = ttk.LabelFrame(parent, text="Simulation Results", padding=6)
        status_frame.pack(fill="x", **pad)
        self._status_var = tk.StringVar(value="—")
        self._warning_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self._status_var, wraplength=280).pack(anchor="w")
        ttk.Label(
            status_frame, textvariable=self._warning_var,
            foreground="red", wraplength=280, font=("TkDefaultFont", 9, "bold")
        ).pack(anchor="w")

    def _toggle_beta_frame(self, frame: ttk.Frame):
        if self._beta_visible.get():
            frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=2)
        else:
            frame.grid_remove()

    def _build_plot_area(self, parent):
        self._fig = Figure(figsize=(9, 10), tight_layout=True)
        self._ax_power = self._fig.add_subplot(3, 1, 1)
        self._ax_rho = self._fig.add_subplot(3, 1, 2)
        self._ax_prec = self._fig.add_subplot(3, 1, 3)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self._canvas, toolbar_frame)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _switch_mode(self):
        self._step_frame.pack_forget()
        self._ramp_frame.pack_forget()
        self._dynamic_frame.pack_forget()
        mode = self._mode_var.get()
        if mode == "step":
            self._step_frame.pack(fill="x")
        elif mode == "ramp":
            self._ramp_frame.pack(fill="x")
        else:
            self._dynamic_frame.pack(fill="x")

    def _refresh_beta_total(self, _=None):
        try:
            vals = [float(v.get()) for v in self._beta_entry_vars]
            self._beta_total_label_var.set(f"β total = {sum(vals):.6f}")
        except ValueError:
            pass

    def _reset_beta_defaults(self):
        for var, v in zip(self._beta_entry_vars, BETA_GROUPS):
            var.set(str(v))
        self._refresh_beta_total()

    def _toggle_log_scale(self):
        scale = "log" if self._log_scale_var.get() else "linear"
        self._ax_power.set_yscale(scale)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _run_simulation(self):
        try:
            Lambda = float(self._lambda_var.get())
            t_end = float(self._t_end_var.get())
            beta_groups = np.array([float(v.get()) for v in self._beta_entry_vars])
            beta_total = float(beta_groups.sum())

            mode = self._mode_var.get()
            frame_map = {"step": self._step_frame, "ramp": self._ramp_frame,
                         "dynamic": self._dynamic_frame}
            profile = frame_map[mode].build_profile(beta_total)

            if Lambda <= 0:
                raise ValueError("Λ must be positive.")
            if t_end <= 0:
                raise ValueError("Simulation duration must be positive.")
            if beta_total <= 0:
                raise ValueError("β total must be positive.")

        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        solver = PKESolver(profile, Lambda, beta_groups, LAMBDA_GROUPS, t_end)
        results = solver.solve()

        if not results.success:
            messagebox.showwarning("Solver Warning", f"Solver reported: {results.message}")

        self._update_plots(results)
        self._update_status(results)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _update_plots(self, r: ResultsBundle):
        PREC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        # --- Plot 1: Neutron Power ---
        ax = self._ax_power
        ax.cla()
        ax.plot(r.t, r.n, color="#1f77b4", linewidth=1.8, label="n(t)/n₀")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Initial (n₀)")
        ax.set_ylabel("Neutron Power  n(t) / n₀")
        ax.set_xlabel("Time [s]")
        ax.set_title("Neutron Power Transient")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        scale = "log" if self._log_scale_var.get() else "linear"
        ax.set_yscale(scale)

        # --- Plot 2: Reactivity ---
        ax2 = self._ax_rho
        ax2.cla()
        # Remove any previous twin axis
        if hasattr(self, "_ax_rho2") and self._ax_rho2 in self._fig.axes:
            self._ax_rho2.remove()

        ax2.plot(r.t, r.rho_dollars, color="#d62728", linewidth=1.8)
        ax2.axhline(1.0, color="darkred", linestyle=":", linewidth=1.0,
                    label="Prompt critical (1 $)")
        ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax2.set_ylabel("Reactivity  ρ  [$]", color="#d62728")
        ax2.set_xlabel("Time [s]")
        ax2.set_title("Reactivity vs. Time")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax2b = ax2.twinx()
        self._ax_rho2 = ax2b
        ax2b.set_ylabel("Reactivity  ρ  [¢]", color="#8b0000")
        cents_min = r.rho_cents.min()
        cents_max = r.rho_cents.max()
        margin = max(abs(cents_max - cents_min) * 0.1, 1.0)
        ax2b.set_ylim(cents_min - margin, cents_max + margin)
        dollars_lim = ax2.get_ylim()
        ax2b.set_ylim(dollars_lim[0] * 100, dollars_lim[1] * 100)

        # --- Plot 3: Precursor Concentrations ---
        ax3 = self._ax_prec
        ax3.cla()
        for i in range(N_GROUPS):
            C_norm = r.C[i] / r.C0[i] if r.C0[i] > 0 else r.C[i]
            ax3.plot(r.t, C_norm, color=PREC_COLORS[i], linewidth=1.2,
                     label=f"Group {i+1}  (λ={LAMBDA_GROUPS[i]} s⁻¹)")
        ax3.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax3.set_ylabel("Cᵢ(t) / Cᵢ(0)")
        ax3.set_xlabel("Time [s]")
        ax3.set_title("Delayed Neutron Precursor Concentrations (normalized)")
        ax3.legend(fontsize=7, loc="upper right", ncol=2)
        ax3.grid(True, alpha=0.3)

        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _update_status(self, r: ResultsBundle):
        period = r.reactor_period
        period_str = f"{period:.2f} s" if period is not None else "∞ (steady/decaying)"
        self._status_var.set(
            f"Period: {period_str}    Max power: {r.max_power:.4f}    "
            f"Final power: {r.final_power:.4f}\n"
            f"β = {r.beta_total:.6f}    Λ = {r.Lambda:.2e} s"
        )
        if r.rho_dollars.max() >= 1.0:
            self._warning_var.set(
                f"⚠  PROMPT SUPERCRITICAL  (peak ρ = {r.rho_dollars.max():.4f} $)"
            )
        else:
            self._warning_var.set("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = PKEApp()
    app.mainloop()


if __name__ == "__main__":
    main()
