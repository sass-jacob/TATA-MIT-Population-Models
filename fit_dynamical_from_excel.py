#!/usr/bin/env python3
"""
Fitting for data from Yash's excel files.

Usage:
  python fit_dynamical_from_excel.py \
    --file /path/to/battery_data.xlsx \
    --epochs 300 \
    --Nc 30 \
    --balance segment \
    --huber_delta_mV 12 \
    --boundary_k 3 \
    --boundary_factor 0.2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Literal, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

###################
# Data Extraction #
###################

@dataclass
class CycleCurve:
    """Container for a single charge or discharge curve."""
    curve_id: int
    cycle_index: int
    step_index: int
    phase: str                 # 'charge' or 'discharge'
    time: np.ndarray           # time in seconds (relative to curve start)
    absolute_time: np.ndarray  # absolute test time in seconds
    voltage: np.ndarray        # voltage in V
    current: np.ndarray        # current in A
    capacity: np.ndarray       # capacity in mAh (relative to curve start)
    soc: np.ndarray            # absolute SOC (0-1, dimensionless)
    
    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.time[-1] - self.time[0] if len(self.time) > 0 else 0.0
    
    @property
    def total_capacity_mAh(self) -> float:
        """Total capacity in mAh."""
        return self.capacity[-1] - self.capacity[0] if len(self.capacity) > 0 else 0.0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        return pd.DataFrame({
            'time_s': self.time,
            'absolute_time_s': self.absolute_time,
            'voltage_V': self.voltage,
            'current_A': self.current,
            'capacity_mAh': self.capacity,
            'soc': self.soc,
            'curve_id': self.curve_id,
            'phase': self.phase,
        })


def load_excel_data(filepath: str) -> pd.DataFrame:
    """Load battery cycling data from Excel file."""
    xl = pd.ExcelFile(filepath)
    
    data_sheet = None
    for sheet in xl.sheet_names:
        if sheet != 'Global_Info':
            data_sheet = sheet
            break
    
    if data_sheet is None:
        raise ValueError("No data sheet found in Excel file")
    
    df = pd.read_excel(xl, sheet_name=data_sheet)
    print(f"Loaded {len(df)} data points from sheet '{data_sheet}'")
    return df


def compute_global_soc(df: pd.DataFrame) -> np.ndarray:
    """Compute global SOC by integrating current over time."""
    time_s = df['Test Time (s)'].values
    current_A = df['Current (A)'].values
    dt = np.diff(time_s, prepend=time_s[0])
    cumulative_capacity_Ah = np.cumsum(current_A * dt / 3600)
    
    min_cap = cumulative_capacity_Ah.min()
    max_cap = cumulative_capacity_Ah.max()
    if max_cap > min_cap:
        global_soc = (cumulative_capacity_Ah - min_cap) / (max_cap - min_cap)
    else:
        global_soc = np.zeros_like(cumulative_capacity_Ah)
    
    return global_soc


def identify_charge_segments(df: pd.DataFrame, 
                             current_threshold: float = 1e-6) -> List[Tuple[int, int]]:
    """Identify contiguous charge segments based on current sign."""
    is_charging = df['Current (A)'] > current_threshold
    charge_diff = is_charging.astype(int).diff().fillna(0)
    
    starts = df.index[charge_diff == 1].tolist()
    ends = df.index[charge_diff == -1].tolist()
    
    if is_charging.iloc[0]:
        starts = [df.index[0]] + starts
    if is_charging.iloc[-1]:
        ends = ends + [df.index[-1] + 1]
    
    return list(zip(starts, ends))


def identify_discharge_segments(df: pd.DataFrame, 
                                current_threshold: float = -1e-6) -> List[Tuple[int, int]]:
    """Identify contiguous discharge segments based on current sign."""
    is_discharging = df['Current (A)'] < current_threshold
    discharge_diff = is_discharging.astype(int).diff().fillna(0)
    
    starts = df.index[discharge_diff == 1].tolist()
    ends = df.index[discharge_diff == -1].tolist()
    
    if is_discharging.iloc[0]:
        starts = [df.index[0]] + starts
    if is_discharging.iloc[-1]:
        ends = ends + [df.index[-1] + 1]
    
    return list(zip(starts, ends))


def extract_curves_from_excel(
    filepath: str,
    min_points: int = 10,
    min_capacity_mAh: float = 0.0,
    charge_current_threshold: float = 1e-6,
    discharge_current_threshold: float = -1e-6,
) -> Tuple[List[CycleCurve], List[CycleCurve]]:
    """
    Extract charge and discharge curves directly from Excel file.
    
    Returns:
        Tuple of (charge_curves, discharge_curves)
    """
    df = load_excel_data(filepath)
    global_soc = compute_global_soc(df)
    df = df.copy()
    df['_global_soc'] = global_soc
    
    def extract_segment_curves(segments, phase: str, capacity_col: str) -> List[CycleCurve]:
        curves = []
        curve_id = 0
        
        for start_idx, end_idx in segments:
            segment = df.loc[start_idx:end_idx-1].copy()
            
            if len(segment) < min_points:
                continue
            
            time_abs = segment['Test Time (s)'].values
            time_rel = time_abs - time_abs[0]
            voltage = segment['Voltage (V)'].values
            current = segment['Current (A)'].values
            soc = segment['_global_soc'].values
            
            # Calculate relative capacity
            if capacity_col in segment.columns:
                capacity_Ah = segment[capacity_col].values
                capacity_mAh = (capacity_Ah - capacity_Ah[0]) * 1000
            else:
                dt_seg = np.diff(time_rel, prepend=0)
                if phase == 'charge':
                    capacity_mAh = np.cumsum(current * dt_seg / 3600 * 1000)
                else:
                    capacity_mAh = np.cumsum(-current * dt_seg / 3600 * 1000)
            
            if capacity_mAh[-1] < min_capacity_mAh:
                continue
            
            cycle_idx = segment['Cycle Index'].iloc[0] if 'Cycle Index' in segment.columns else 0
            step_idx = segment['Step Index'].iloc[0] if 'Step Index' in segment.columns else 0
            
            curves.append(CycleCurve(
                curve_id=curve_id,
                cycle_index=int(cycle_idx),
                step_index=int(step_idx),
                phase=phase,
                time=time_rel,
                absolute_time=time_abs,
                voltage=voltage,
                current=current,
                capacity=capacity_mAh,
                soc=soc,
            ))
            curve_id += 1
        
        return curves
    
    # Extract charge curves
    charge_segments = identify_charge_segments(df, charge_current_threshold)
    charge_curves = extract_segment_curves(charge_segments, 'charge', 'Charge Capacity (Ah)')
    print(f"Extracted {len(charge_curves)} charge curves")
    
    # Extract discharge curves
    discharge_segments = identify_discharge_segments(df, discharge_current_threshold)
    discharge_curves = extract_segment_curves(discharge_segments, 'discharge', 'Discharge Capacity (Ah)')
    print(f"Extracted {len(discharge_curves)} discharge curves")
    
    return charge_curves, discharge_curves


def curves_to_dataframe(charge_curves: List[CycleCurve], 
                        discharge_curves: List[CycleCurve]) -> pd.DataFrame:
    """Convert curves to a combined DataFrame for fitting."""
    dfs = []
    
    for curve in charge_curves:
        df = curve.to_dataframe()
        dfs.append(df)
    
    for curve in discharge_curves:
        df = curve.to_dataframe()
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No curves extracted from the Excel file")
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


################
# Fitting Code #
################
def load_and_stitch_experiment_from_df(
    df_combined: pd.DataFrame,
    max_cycle_id: int | None = None,
) -> pd.DataFrame:
    """
    Prepare the combined DataFrame for fitting.
    Expects columns: absolute_time_s, voltage_V, current_A, capacity_mAh, curve_id, phase, time_s
    """
    df = df_combined.copy()
    df = df.sort_values("absolute_time_s").reset_index(drop=True)

    if max_cycle_id is not None:
        df = df[df["curve_id"] <= max_cycle_id].reset_index(drop=True)

    needed = {"absolute_time_s", "voltage_V", "current_A", "capacity_mAh", "curve_id", "phase", "time_s"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def capacity_from_cycle0(df_all: pd.DataFrame) -> float:
    df0 = df_all[df_all["curve_id"] == 0]
    if len(df0) == 0:
        raise ValueError("No curve_id == 0 found; cannot define total capacity from first cycle.")

    q_charge = df0[df0["phase"] == "charge"]["capacity_mAh"].max()
    q_dis = df0[df0["phase"] == "discharge"]["capacity_mAh"].max()

    q_charge = float(q_charge) if np.isfinite(q_charge) else 0.0
    q_dis = float(q_dis) if np.isfinite(q_dis) else 0.0

    q_total_mAh = max(q_charge, q_dis)
    if q_total_mAh <= 0:
        raise ValueError("Cycle 0 capacity appears invalid (<=0).")

    return q_total_mAh / 1000.0  # Ah


def coulomb_count_soc_reference(df_all: pd.DataFrame, Q_total_Ah: float) -> np.ndarray:
    """
    Coulomb-count SOC reference using segment-aware rest handling.
    """
    t_s = df_all["absolute_time_s"].to_numpy(dtype=np.float64)
    I_A = df_all["current_A"].to_numpy(dtype=np.float64)
    curve = df_all["curve_id"].to_numpy(dtype=np.int64)
    phase = df_all["phase"].to_numpy(dtype=object)

    dt_h = np.diff(t_s) / 3600.0
    seg_change = (curve[1:] != curve[:-1]) | (phase[1:] != phase[:-1])

    # Treat gaps at segment change as rest: I=0 during that interval
    I_step = np.where(seg_change, 0.0, I_A[:-1])

    cum_Ah = np.concatenate([[0.0], np.cumsum(I_step * dt_h)])
    cum_min = float(cum_Ah.min())
    soc_raw = (cum_Ah - cum_min) / float(Q_total_Ah)

    eps = 1e-4
    soc = soc_raw * (1.0 - 2.0 * eps) + eps
    # clip soc to be a valid range
    return np.clip(soc, eps, 1.0 - eps)


def build_fit_arrays(
    df_all: pd.DataFrame,
    Q_total_Ah: float,
    exclude_cycle0: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Build time series arrays for fitting.
    """
    soc_ref = coulomb_count_soc_reference(df_all, Q_total_Ah)
    df_all = df_all.copy()
    df_all["soc_anode_ref"] = soc_ref

    if exclude_cycle0:
        df = df_all[df_all["curve_id"] > 0].reset_index(drop=True)
    else:
        df = df_all.reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Too few points after filtering.")

    t_abs = df["absolute_time_s"].to_numpy(dtype=np.float64)
    t_h = (t_abs - t_abs[0]) / 3600.0

    V_meas = df["voltage_V"].to_numpy(dtype=np.float64)

    # Current normalized by Cycle0 capacity => "C-rate-like" [1/h]
    I_c_rate = df["current_A"].to_numpy(dtype=np.float64) / float(Q_total_Ah)

    # Model sign convention (cathode-like)
    I_model = -I_c_rate

    curve = df["curve_id"].to_numpy(dtype=np.int64)
    phase = df["phase"].to_numpy(dtype=object)

    # Segment boundaries (curve_id or phase changes)
    seg_change = (curve[1:] != curve[:-1]) | (phase[1:] != phase[:-1])

    segment_id = np.zeros(len(df), dtype=np.int64)
    segment_id[1:] = np.cumsum(seg_change).astype(np.int64)

    segment_start = np.zeros(len(df), dtype=np.bool_)
    segment_start[0] = True
    segment_start[1:] = seg_change

    # dt for advance i->i+1
    dt_steps = np.diff(t_h)
    dt_steps = np.where(dt_steps <= 0.0, 1e-6, dt_steps)

    # current for advance i->i+1
    I_steps = np.where(seg_change, 0.0, I_model[:-1])

    # after segment change, re-equilibrate at i+1 to I_model[i+1]
    need_eq = seg_change.astype(np.bool_)

    soc_anode_ref_fit = df["soc_anode_ref"].to_numpy(dtype=np.float64)

    return dict(
        t_h=t_h,
        V_meas=V_meas,
        I_model=I_model,
        dt_steps=dt_steps,
        I_steps=I_steps,
        need_eq=need_eq,
        soc_anode_ref=soc_anode_ref_fit,
        segment_id=segment_id,
        segment_start=segment_start,
        curve_id=curve,
        phase=phase,
    )


def compute_loss_weights(
    *,
    V_meas: np.ndarray,
    segment_id: np.ndarray,
    curve_id: np.ndarray,
    balance: Literal["none", "segment", "cycle"] = "segment",
    boundary_start: np.ndarray | None = None,
    boundary_k: int = 0,
    boundary_factor: float = 1.0,
) -> np.ndarray:
    """
    Produce per-sample weights w[i] to prevent early/segment-start points dominating.
    """
    N = len(V_meas)
    w = np.ones(N, dtype=np.float64)

    if balance == "segment":
        counts = np.bincount(segment_id)
        w = 1.0 / counts[segment_id]
    elif balance == "cycle":
        cid = curve_id.astype(np.int64)
        uniq, inv = np.unique(cid, return_inverse=True)
        counts = np.bincount(inv)
        w = 1.0 / counts[inv]
    elif balance == "none":
        w[:] = 1.0
    else:
        raise ValueError(f"Unknown balance={balance}")

    # Downweight early transient points at segment starts
    if boundary_start is not None and boundary_k > 0 and boundary_factor < 1.0:
        start_idxs = np.flatnonzero(boundary_start)
        for s in start_idxs:
            e = min(N, s + boundary_k)
            w[s:e] *= boundary_factor

    # Normalize so average weight ~ 1
    w *= (N / np.sum(w))
    return w


@dataclass(frozen=True)
class FitConfig:
    Nc: int = 30
    max_newton_iters: int = 8
    max_v_iters: int = 8
    newton_damping: float = 1.0
    upwind_smooth: float = 0.0


@dataclass(frozen=True)
class PhysicalParams:
    omega: jnp.ndarray
    k0: jnp.ndarray
    D0: jnp.ndarray
    phi_ref: jnp.ndarray
    R0: jnp.ndarray
    V_T: jnp.ndarray
    Q: jnp.ndarray


def unpack_params(theta_raw: jnp.ndarray) -> Tuple[PhysicalParams, jnp.ndarray]:
    log_omega, log_k0, log_D0, log_phi_ref, log_R0, soc0_raw = theta_raw

    omega = jnp.exp(log_omega)
    k0 = jnp.exp(log_k0)
    D0 = jnp.exp(log_D0)
    phi_ref = jnp.exp(log_phi_ref)
    R0 = jnp.exp(log_R0)

    eps = 1e-4
    soc0 = eps + (1.0 - 2.0 * eps) * jax.nn.sigmoid(soc0_raw)

    p = PhysicalParams(
        omega=omega,
        k0=k0,
        D0=D0,
        phi_ref=phi_ref,
        R0=R0,
        V_T=jnp.array(0.0257, dtype=jnp.float64),
        Q=jnp.array(1.0, dtype=jnp.float64),
    )
    return p, soc0


@jax.jit
def chemical_potential(c: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(c / (1.0 - c)) + omega * (1.0 - 2.0 * c)


@jax.jit
def exchange_current_density(c: jnp.ndarray, k0: jnp.ndarray) -> jnp.ndarray:
    return k0 * jnp.sqrt(c) * (1.0 - c)


@jax.jit
def reaction_rate(
    c: jnp.ndarray,
    omega: jnp.ndarray,
    k0: jnp.ndarray,
    phi_ref: jnp.ndarray,
    V_T: jnp.ndarray,
    V: jnp.ndarray,
) -> jnp.ndarray:
    mu = chemical_potential(c, omega)
    j0 = exchange_current_density(c, k0)
    delta_mu = (V - phi_ref) / V_T + mu
    return -j0 * delta_mu


def finite_volume_fluxes(
    f: jnp.ndarray,
    R: jnp.ndarray,
    D: jnp.ndarray,
    dc: jnp.ndarray,
    upwind_smooth: float,
) -> jnp.ndarray:
    R_faces = 0.5 * (R[:-1] + R[1:])

    if upwind_smooth and upwind_smooth > 0.0:
        theta = 0.5 * (1.0 + jnp.tanh(upwind_smooth * R_faces))
        f_upwind = theta * f[:-1] + (1.0 - theta) * f[1:]
    else:
        f_upwind = jnp.where(R_faces >= 0.0, f[:-1], f[1:])

    flux_adv = R_faces * f_upwind
    flux_dif = -D * (f[1:] - f[:-1]) / dc
    return flux_adv + flux_dif


def build_model_grids(Nc: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    c_grid = jnp.linspace(1e-5, 1.0 - 1e-5, Nc, dtype=jnp.float64)
    dc = c_grid[1] - c_grid[0]
    return c_grid, dc


def current_constraint_integral(
    f: jnp.ndarray,
    V: jnp.ndarray,
    I_app: jnp.ndarray,
    p: PhysicalParams,
    c_grid: jnp.ndarray,
    dc: jnp.ndarray,
    cfg: FitConfig,
) -> jnp.ndarray:
    R = reaction_rate(c_grid, p.omega, p.k0, p.phi_ref, p.V_T, V)
    J_faces = finite_volume_fluxes(f, R, p.D0, dc, cfg.upwind_smooth)
    J_faces = jnp.pad(J_faces, (1, 1), constant_values=0.0)
    int_J = jnp.sum(J_faces) * dc
    return int_J - (I_app / p.Q)


def equilibrate_voltage(
    f: jnp.ndarray,
    V_guess: jnp.ndarray,
    I_app: jnp.ndarray,
    p: PhysicalParams,
    c_grid: jnp.ndarray,
    dc: jnp.ndarray,
    cfg: FitConfig,
) -> jnp.ndarray:
    def g(V):
        return current_constraint_integral(f, V, I_app, p, c_grid, dc, cfg)

    dg = jax.grad(g)

    def body(i, V):
        gv = g(V)
        dgv = dg(V)
        step = gv / (dgv + 1e-14)
        step = jnp.clip(step, -0.25, 0.25)
        return V - step

    return jax.lax.fori_loop(0, cfg.max_v_iters, body, V_guess)


def residual_backward_euler(
    y: jnp.ndarray,
    y_old: jnp.ndarray,
    dt: jnp.ndarray,
    I_app: jnp.ndarray,
    p: PhysicalParams,
    c_grid: jnp.ndarray,
    dc: jnp.ndarray,
    cfg: FitConfig,
) -> jnp.ndarray:
    Nc = c_grid.shape[0]
    f = y[:Nc]
    V = y[Nc]
    f_old = y_old[:Nc]

    R = reaction_rate(c_grid, p.omega, p.k0, p.phi_ref, p.V_T, V)
    J_faces = finite_volume_fluxes(f, R, p.D0, dc, cfg.upwind_smooth)
    J_faces = jnp.pad(J_faces, (1, 1), constant_values=0.0)

    div_J = (J_faces[1:] - J_faces[:-1]) / dc
    res_f = (f - f_old) / dt + div_J

    int_J = jnp.sum(J_faces) * dc
    res_V = int_J - (I_app / p.Q)

    return jnp.concatenate([res_f, jnp.array([res_V], dtype=jnp.float64)])


jac_y = jax.jacfwd(residual_backward_euler, argnums=0)


def newton_solve_step(
    y_init: jnp.ndarray,
    y_old: jnp.ndarray,
    dt: jnp.ndarray,
    I_app: jnp.ndarray,
    p: PhysicalParams,
    c_grid: jnp.ndarray,
    dc: jnp.ndarray,
    cfg: FitConfig,
) -> jnp.ndarray:
    def body(i, y):
        F = residual_backward_euler(y, y_old, dt, I_app, p, c_grid, dc, cfg)
        J = jac_y(y, y_old, dt, I_app, p, c_grid, dc, cfg)
        delta = jnp.linalg.solve(J, -F)
        return y + cfg.newton_damping * delta

    return jax.lax.fori_loop(0, cfg.max_newton_iters, body, y_init)


def initialize_distribution(
    c_grid: jnp.ndarray,
    dc: jnp.ndarray,
    soc0: jnp.ndarray,
    sigma: float = 0.05
) -> jnp.ndarray:
    f = jnp.exp(-0.5 * ((c_grid - soc0) / sigma) ** 2)
    return f / (jnp.sum(f) * dc)


def simulate_voltage_trace(
    theta_raw: jnp.ndarray,
    V_meas: jnp.ndarray,
    I_meas: jnp.ndarray,
    dt_steps: jnp.ndarray,
    I_steps: jnp.ndarray,
    need_eq: jnp.ndarray,
    c_grid: jnp.ndarray,
    dc: jnp.ndarray,
    cfg: FitConfig,
) -> jnp.ndarray:
    p, soc0 = unpack_params(theta_raw)

    f0 = initialize_distribution(c_grid, dc, soc0, sigma=0.05)

    V0_guess = p.phi_ref
    V0 = equilibrate_voltage(f0, V0_guess, I_meas[0], p, c_grid, dc, cfg)
    y0 = jnp.concatenate([f0, jnp.array([V0], dtype=jnp.float64)])

    Vpred0 = V0 + I_meas[0] * p.R0

    def scan_body(y, inp):
        dt, I_adv, I_next, do_eq = inp

        y_next = newton_solve_step(y, y, dt, I_adv, p, c_grid, dc, cfg)

        Nc = c_grid.shape[0]
        f_next = y_next[:Nc]
        V_next = y_next[Nc]

        def _do_equilibrate(_):
            V_eq = equilibrate_voltage(f_next, V_next, I_next, p, c_grid, dc, cfg)
            return jnp.concatenate([f_next, jnp.array([V_eq], dtype=jnp.float64)])

        y_eq = jax.lax.cond(do_eq, _do_equilibrate, lambda _: y_next, operand=None)

        V_int = y_eq[Nc]
        V_term = V_int + I_next * p.R0
        return y_eq, V_term

    inputs = (dt_steps, I_steps, I_meas[1:], need_eq)
    _, Vpred_rest = jax.lax.scan(scan_body, y0, inputs)

    return jnp.concatenate([jnp.array([Vpred0]), Vpred_rest])


@jax.jit
def pseudo_huber(r: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    z = r / (delta + 1e-30)
    return (delta ** 2) * (jnp.sqrt(1.0 + z * z) - 1.0)


def make_loss_fn(
    V_meas: jnp.ndarray,
    I_meas: jnp.ndarray,
    dt_steps: jnp.ndarray,
    I_steps: jnp.ndarray,
    need_eq: jnp.ndarray,
    weights: jnp.ndarray,
    huber_delta_V: float,
    c_grid: jnp.ndarray,
    dc: jnp.ndarray,
    cfg: FitConfig,
):
    delta = jnp.array(huber_delta_V, dtype=jnp.float64)

    @jax.jit
    def loss(theta_raw: jnp.ndarray) -> jnp.ndarray:
        V_pred = simulate_voltage_trace(
            theta_raw, V_meas, I_meas, dt_steps, I_steps, need_eq, c_grid, dc, cfg
        )
        r = V_pred - V_meas

        per = pseudo_huber(r, delta)
        L = jnp.sum(weights * per) / (jnp.sum(weights) + 1e-30)

        p, _ = unpack_params(theta_raw)
        reg = 0.0
        reg += 1e-3 * (jnp.log(p.D0) - jnp.log(1e-1)) ** 2
        reg += 1e-4 * (jnp.log(p.k0) - jnp.log(4.3)) ** 2
        reg += 1e-4 * (p.phi_ref - 3.4) ** 2
        reg += 1e-4 * (jnp.log(p.R0) - jnp.log(0.04)) ** 2

        out = L + reg
        out = jnp.nan_to_num(out, nan=1e6, posinf=1e6, neginf=1e6)
        return out

    return loss


def main():
    parser = argparse.ArgumentParser(
        description='Fit dynamical battery model directly from Excel data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--excel_file", type=str, required=True,
                        help="Path to Excel file with battery cycling data")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--Nc", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max_cycle_id", type=int, default=None)

    parser.add_argument("--max_newton_iters", type=int, default=8)
    parser.add_argument("--max_v_iters", type=int, default=8)
    parser.add_argument("--newton_damping", type=float, default=1.0)
    parser.add_argument("--upwind_smooth", type=float, default=0.0)

    # Extraction parameters
    parser.add_argument("--min_points", type=int, default=10,
                        help="Minimum data points for a valid curve")
    parser.add_argument("--min_capacity", type=float, default=0.0,
                        help="Minimum capacity (mAh) for a valid curve")
    parser.add_argument("--charge_current_threshold", type=float, default=1e-6,
                        help="Current threshold for charge detection (A)")
    parser.add_argument("--discharge_current_threshold", type=float, default=-1e-6,
                        help="Current threshold for discharge detection (A)")

    # Robust + weighting controls
    parser.add_argument("--huber_delta_mV", type=float, default=12.0,
                        help="Pseudo-Huber delta in mV. This controls how to weight the loss when the deltaV is large.")
    parser.add_argument("--balance", type=str, default="segment",
                        choices=["none", "segment", "cycle"],
                        help="How to balance contributions: none|segment|cycle.")
    parser.add_argument("--boundary_k", type=int, default=3,
                        help="Downweight first K samples of each segment.")
    parser.add_argument("--boundary_factor", type=float, default=0.2,
                        help="Factor (<1) applied to first K samples of each segment.")
    
    # Output options
    parser.add_argument("--output_prefix", type=str, default="fit_result",
                        help="Prefix for output files")
    
    args = parser.parse_args()

    # Extract curves directly from Excel
    print(f"\n{'='*60}")
    print("STEP 1: Extracting curves from Excel file")
    print(f"{'='*60}")
    print(f"Loading: {args.excel_file}")
    
    charge_curves, discharge_curves = extract_curves_from_excel(
        args.excel_file,
        min_points=args.min_points,
        min_capacity_mAh=args.min_capacity,
        charge_current_threshold=args.charge_current_threshold,
        discharge_current_threshold=args.discharge_current_threshold,
    )

    # Convert to DataFrame for fitting
    print(f"\n{'='*60}")
    print("STEP 2: Preparing data for fitting")
    print(f"{'='*60}")
    
    df_combined = curves_to_dataframe(charge_curves, discharge_curves)
    df_all = load_and_stitch_experiment_from_df(df_combined, max_cycle_id=args.max_cycle_id)

    Q_total_Ah = capacity_from_cycle0(df_all)
    print(f"Total capacity from cycle0: {Q_total_Ah*1000:.6f} mAh ({Q_total_Ah:.9f} Ah)")

    arrays = build_fit_arrays(df_all, Q_total_Ah, exclude_cycle0=True)

    t_h = arrays["t_h"]
    V_meas_np = arrays["V_meas"]
    I_model_np = arrays["I_model"]
    dt_steps_np = arrays["dt_steps"]
    I_steps_np = arrays["I_steps"]
    need_eq_np = arrays["need_eq"]
    soc_anode_np = arrays["soc_anode_ref"]

    segment_id = arrays["segment_id"]
    segment_start = arrays["segment_start"]
    curve_id = arrays["curve_id"]

    # Initial SOC guess (cathode convention)
    soc0_guess = float(1.0 - soc_anode_np[0])

    print(f"Fit points (excluding cycle0): {len(V_meas_np)}")
    print(f"Time range: {t_h.min():.3f}h -> {t_h.max():.3f}h")
    print(f"Current range (model, 1/h): {I_model_np.min():.4g} -> {I_model_np.max():.4g}")
    print(f"Initial SOC guess (cathode): {soc0_guess:.6f}")

    # Loss weights
    w_np = compute_loss_weights(
        V_meas=V_meas_np,
        segment_id=segment_id,
        curve_id=curve_id,
        balance=args.balance,
        boundary_start=segment_start,
        boundary_k=args.boundary_k,
        boundary_factor=args.boundary_factor,
    )
    print(f"Weights: min={w_np.min():.3g}, max={w_np.max():.3g}, mean={w_np.mean():.3g}")

    # Build JAX arrays
    print(f"\n{'='*60}")
    print("STEP 3: Running optimization")
    print(f"{'='*60}")
    
    cfg = FitConfig(
        Nc=args.Nc,
        max_newton_iters=args.max_newton_iters,
        max_v_iters=args.max_v_iters,
        newton_damping=args.newton_damping,
        upwind_smooth=args.upwind_smooth,
    )
    c_grid, dc = build_model_grids(cfg.Nc)

    V_meas = jnp.array(V_meas_np, dtype=jnp.float64)
    I_meas = jnp.array(I_model_np, dtype=jnp.float64)
    dt_steps = jnp.array(dt_steps_np, dtype=jnp.float64)
    I_steps = jnp.array(I_steps_np, dtype=jnp.float64)
    need_eq = jnp.array(need_eq_np, dtype=jnp.bool_)
    weights = jnp.array(w_np, dtype=jnp.float64)

    # Initialize parameters
    def logit(x):
        x = np.clip(x, 1e-4, 1 - 1e-4)
        return np.log(x / (1 - x))

    theta0 = np.array([
        np.log(5.3),        # omega
        np.log(4.3),        # k0
        np.log(1e-1),       # D0
        np.log(3.42),       # phi_ref
        np.log(0.04),       # R0
        logit(soc0_guess),  # soc0_raw
    ], dtype=np.float64)
    theta = jnp.array(theta0)

    huber_delta_V = args.huber_delta_mV / 1000.0

    loss_fn = make_loss_fn(
        V_meas, I_meas, dt_steps, I_steps, need_eq,
        weights=weights,
        huber_delta_V=huber_delta_V,
        c_grid=c_grid, dc=dc, cfg=cfg,
    )
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    schedule = optax.exponential_decay(
        init_value=args.lr,
        transition_steps=50,
        decay_rate=0.95,
        staircase=False,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule),
    )
    opt_state = tx.init(theta)

    # Train
    print("\nStarting optimization...")
    losses = []

    for epoch in range(args.epochs):
        loss_val, grads = value_and_grad(theta)
        updates, opt_state = tx.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        losses.append(float(loss_val))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            p, soc0 = unpack_params(theta)
            print(
                f"Epoch {epoch:4d} | loss={float(loss_val):.6e} | "
                f"omega={float(p.omega):.4g} k0={float(p.k0):.4g} D0={float(p.D0):.4g} "
                f"phi_ref={float(p.phi_ref):.4f} R0={float(p.R0):.4g} soc0={float(soc0):.4f}"
            )

    # Final simulation and plot
    print(f"\n{'='*60}")
    print("STEP 4: Generating results")
    print(f"{'='*60}")
    
    p_final, soc0_final = unpack_params(theta)
    V_pred = simulate_voltage_trace(theta, V_meas, I_meas, dt_steps, I_steps, need_eq, c_grid, dc, cfg)
    V_pred_np = np.array(V_pred)

    rmse_mV = 1000.0 * float(np.sqrt(np.mean((V_pred_np - V_meas_np) ** 2)))
    print(f"\nFinal RMSE: {rmse_mV:.2f} mV")
    print("Final parameters:")
    final_params = {
        "omega": float(p_final.omega),
        "k0": float(p_final.k0),
        "D0": float(p_final.D0),
        "phi_ref": float(p_final.phi_ref),
        "R0": float(p_final.R0),
        "soc0_cathode": float(soc0_final),
    }
    print(final_params)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2)

    axV = fig.add_subplot(gs[:, 0])
    axL = fig.add_subplot(gs[0, 1])
    axI = fig.add_subplot(gs[1, 1])

    axV.plot(t_h, V_meas_np, linewidth=1.0, label="Measured")
    axV.plot(t_h, V_pred_np, linewidth=1.5, label="Model")
    axV.set_xlabel("Time [h]")
    axV.set_ylabel("Voltage [V]")
    axV.set_title(f"Voltage fit (RMSE {rmse_mV:.1f} mV)")
    axV.grid(True, alpha=0.3)
    axV.legend()

    axL.plot(losses)
    axL.set_title("Training loss (robust)")
    axL.set_xlabel("Epoch")
    axL.set_ylabel("Loss")
    axL.grid(True, alpha=0.3)

    axI.plot(t_h, I_model_np)
    axI.set_title("Applied current (model sign convention)")
    axI.set_xlabel("Time [h]")
    axI.set_ylabel("I [1/h]")
    axI.grid(True, alpha=0.3)

    plt.tight_layout()
    output_plot = f"{args.output_prefix}_from_excel.png"
    plt.savefig(output_plot, dpi=160)
    print(f"Saved plot: {output_plot}")


if __name__ == "__main__":
    main()
