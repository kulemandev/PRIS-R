from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPISODE_S = 0.250

EPISODE_COLUMNS = [
    "step",
    "thr_kbps_critical_DL", "thr_kbps_critical_UL",
    "thr_kbps_performance_DL", "thr_kbps_performance_UL",
    "violation_rate_pct_critical_DL", "violation_rate_pct_critical_UL",
    "violation_rate_pct_performance_DL", "violation_rate_pct_performance_UL",
    "violation_gap_rate_critical_DL", "violation_gap_rate_critical_UL",
    "violation_gap_rate_performance_DL", "violation_gap_rate_performance_UL",
    "punctured_prbs_DL", "punctured_prbs_UL",
    "runtime_ms", "ram_mb", "energy_j",
    "emergency_mode",
]

JOB_COLUMNS = [
    "train_id", "app", "link", "slice", "seq",
    "arrival_time_s", "first_served_time_s", "done_time_s",
    "delay_ms", "total_bits",
]

OPEN_COLUMNS = [
    "train_id", "app", "link", "slice", "seq",
    "arrival_time_s", "first_served_time_s", "age_s_at_end",
    "remaining_bits", "total_bits",
]

APP_DEFS = {
    "voice": dict(slice="critical"),
    "etcs": dict(slice="critical"),
    "ato": dict(slice="critical"),
    "remote_engine_ctrl": dict(slice="critical"),
    "pub_warn": dict(slice="critical"),
    "telemetry_nc": dict(slice="performance"),
    "equip_ctl": dict(slice="performance"),
    "asset_tel": dict(slice="performance"),
    "pis": dict(slice="performance"),
    "video_surv": dict(slice="performance"),
}

APP_PACKET_PROFILE = {
    "voice": {"pdb_ms": 65.0},
    "etcs": {"pdb_ms": 50.0},
    "ato": {"pdb_ms": 50.0},
    "remote_engine_ctrl": {"pdb_ms": 50.0},
    "pub_warn": {"pdb_ms": 65.0},
    "telemetry_nc": {"pdb_ms": 190.0},
    "equip_ctl": {"pdb_ms": 50.0},
    "asset_tel": {"pdb_ms": 190.0},
    "pis": {"pdb_ms": 280.0},
    "video_surv": {"pdb_ms": 280.0},
}

APP_MIN_RELIABILITY_PCT = {
    "voice": 99.9,
    "etcs": 99.999,
    "ato": 99.999,
    "remote_engine_ctrl": 99.999,
    "pub_warn": 99.99,
    "telemetry_nc": 99.999,
    "equip_ctl": 99.999,
    "asset_tel": 99.99,
    "pis": np.nan,
    "video_surv": 97.5,
}

PLOT_DPI = 220
PLOT_FIGSIZE_LINE = (12, 7)
PLOT_FIGSIZE_BAR  = (13, 7)
PLOT_FIGSIZE_REL  = (15, 7.5)
FONT_TITLE    = 20
FONT_LABEL    = 24
FONT_TICK     = 15
FONT_LEGEND   = 18
FONT_BAR_LABEL = 18
SAVEFIG_KW = {"bbox_inches": "tight", "pad_inches": 0.02}


def reliability_pct_to_nines(rel_pct: np.ndarray) -> np.ndarray:
    rel = np.asarray(rel_pct, dtype=float)
    ok  = np.isfinite(rel)
    out = np.full(rel.shape, np.nan, dtype=float)
    if np.any(ok):
        frac    = np.clip(rel[ok] / 100.0, 0.0, 1.0 - 1e-12)
        out[ok] = -np.log10(np.clip(1.0 - frac, 1e-12, None))
    return out


def read_csv_safe(path: Path, required_columns: List[str]) -> pd.DataFrame:
    if (not path.exists()) or path.stat().st_size == 0:
        return pd.DataFrame(columns=required_columns)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=required_columns)
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def episode_delay_timeseries(
    job_df: pd.DataFrame, steps: np.ndarray, slice_name: str, link: str
) -> pd.DataFrame:
    if job_df.empty:
        return pd.DataFrame({"step": steps, "delay_ms": np.nan})
    job_df = job_df.copy()
    job_df = job_df[(job_df["slice"] == slice_name) & (job_df["link"] == link)]
    if job_df.empty:
        return pd.DataFrame({"step": steps, "delay_ms": np.nan})
    job_df["done_time_s"] = pd.to_numeric(job_df["done_time_s"], errors="coerce")
    job_df = job_df.dropna(subset=["done_time_s"]).copy()
    if job_df.empty:
        return pd.DataFrame({"step": steps, "delay_ms": np.nan})
    job_df["done_step"] = np.floor(job_df["done_time_s"] / EPISODE_S).astype(int)
    g   = job_df.groupby("done_step")["delay_ms"].mean().reset_index()
    out = (pd.DataFrame({"step": steps})
           .merge(g, left_on="step", right_on="done_step", how="left")
           .drop(columns=["done_step"]))
    return out.sort_values("step").reset_index(drop=True)


def plot_compare(
    reac_ep: pd.DataFrame, reac_jobs: pd.DataFrame, reac_open: pd.DataFrame,
    heur_ep: pd.DataFrame, heur_jobs: pd.DataFrame, heur_open: pd.DataFrame,
    pro_ep:  pd.DataFrame, pro_jobs:  pd.DataFrame, pro_open:  pd.DataFrame,
    drl_ep:  pd.DataFrame, drl_jobs:  pd.DataFrame, drl_open:  pd.DataFrame,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── four output subfolders ───────────────────────────────────────────────
    # throughput/   served kbps + throughput violation rate (% demand not met)
    # delay/        packet delay (ms) + deadline-based delay violation rate
    # reliability/  QoS reliability = full delivery before deadline
    # overhead/     PRB puncturing, runtime, RAM, energy, energy efficiency
    thr_dir         = out_dir / "throughput"
    delay_dir       = out_dir / "delay"
    reliability_dir = out_dir / "reliability"
    overhead_dir    = out_dir / "overhead"
    for d in [thr_dir, delay_dir, reliability_dir, overhead_dir]:
        d.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "axes.titlesize":  FONT_TITLE,
        "axes.labelsize":  FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    })

    methods_all = [
        ("Reactive",           reac_ep, reac_jobs, reac_open),
        ("Proactive",  heur_ep, heur_jobs, heur_open),
        ("ProReactive_heuristic",     pro_ep,  pro_jobs,  pro_open),
        ("OurPRISR_ProReactive_DRL", drl_ep,  drl_jobs,  drl_open),
    ]
    methods = [(name, ep_df, job_df, open_df) for name, ep_df, job_df, open_df in methods_all if not ep_df.empty]
    if not methods:
        raise ValueError("No simulation outputs with episode data were found to plot.")

    step_sets = [set(pd.to_numeric(ep_df["step"], errors="coerce").dropna().astype(int).unique()) for _, ep_df, _, _ in methods]
    if len(step_sets) == 1:
        steps = np.array(sorted(step_sets[0]))
    else:
        steps = np.array(sorted(set.intersection(*step_sets)))
    if len(steps) == 0:
        steps = np.array(sorted(set.union(*step_sets)))

    m_eps:  Dict[str, pd.DataFrame] = {}
    m_jobs: Dict[str, pd.DataFrame] = {}
    m_open: Dict[str, pd.DataFrame] = {}
    for name, ep_df, job_df, open_df in methods:
        m_eps[name]  = ep_df[ep_df["step"].isin(steps)].copy()
        m_jobs[name] = job_df.copy()
        m_open[name] = open_df.copy()

    # ── generic plot helpers ─────────────────────────────────────────────────

    def _line_plot(xs_by_name, ys_by_name, xlabel, ylabel, title, save_path):
        plt.figure(figsize=PLOT_FIGSIZE_LINE)
        for name, *_ in methods:
            plt.plot(xs_by_name[name], ys_by_name[name], label=name)
        plt.xlabel(xlabel, fontsize=FONT_LABEL)
        plt.ylabel(ylabel, fontsize=FONT_LABEL)
        plt.legend(fontsize=FONT_LEGEND)
        plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
        plt.tight_layout(pad=0.2)
        plt.savefig(save_path, dpi=PLOT_DPI, **SAVEFIG_KW)
        plt.close()

    def _bar_group_plot(labels, metrics_by_method, ylabel, title, save_path, bar_fmt="%.2f"):
        def _fmt_bar_value(v: float) -> str:
            if not np.isfinite(v):
                return "NA"
            if abs(v) <= 5e-12:
                return "0"
            return bar_fmt % v

        names = list(metrics_by_method.keys())
        x       = np.arange(len(labels))
        w       = 0.18
        offsets = np.linspace(-1.5 * w, 1.5 * w, len(names))
        plt.figure(figsize=PLOT_FIGSIZE_BAR)
        ax = plt.gca()
        all_vals = []
        for name, off in zip(names, offsets):
            vals = [metrics_by_method[name].get(lbl, 0.0) for lbl in labels]
            all_vals.extend([float(v) for v in vals if np.isfinite(v)])
            bars = ax.bar(x + off, vals, width=w, label=name)
            ax.bar_label(
                bars,
                labels=[_fmt_bar_value(v) for v in vals],
                padding=4,
                fontsize=FONT_BAR_LABEL,
                rotation=90,
            )
        plt.xticks(x, labels, fontsize=FONT_TICK)
        plt.yticks(fontsize=FONT_TICK)
        ymax = max(all_vals) if all_vals else 1.0
        ax.set_ylim(0.0, ymax + max(2.0, 0.18 * ymax))
        plt.ylabel(ylabel, fontsize=FONT_LABEL)
        plt.legend(fontsize=FONT_LEGEND)
        plt.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
        plt.tight_layout(pad=0.2)
        plt.savefig(save_path, dpi=PLOT_DPI, **SAVEFIG_KW)
        plt.close()

    def safe_mean(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce")
        return float(np.nanmean(s.values)) if len(s) else 0.0

    def _subset_by_mode(ep_df, job_df, mode_flag):
        if "emergency_mode" in ep_df.columns:
            ep_sub = ep_df[ep_df["emergency_mode"] == mode_flag].copy()
        else:
            ep_sub = ep_df.copy() if mode_flag == 0 else ep_df.iloc[0:0].copy()
        step_set = set(pd.to_numeric(ep_sub["step"], errors="coerce").dropna().astype(int)) if not ep_sub.empty else set()
        if job_df.empty or not step_set:
            return ep_sub, job_df.iloc[0:0].copy()
        jb = job_df.copy()
        done_s = pd.to_numeric(jb.get("done_time_s", np.nan), errors="coerce")
        jb["_done_step"] = np.floor((done_s - 1e-12) / EPISODE_S)
        jb = jb.dropna(subset=["_done_step"]).copy()
        jb["_done_step"] = jb["_done_step"].astype(int)
        return ep_sub, jb[jb["_done_step"].isin(step_set)].drop(columns=["_done_step"], errors="ignore")

    mode_specs = [
        (0, "normal",    "Normal"),
        (1, "emergency", "Emergency"),
    ]

    # ════════════════════════════════════════════════════════════════════════
    # throughput/
    # ════════════════════════════════════════════════════════════════════════

    for slice_name, link, fn in [
        ("critical",    "DL", "thr_critical_DL.png"),
        ("critical",    "UL", "thr_critical_UL.png"),
        ("performance", "DL", "thr_performance_DL.png"),
        ("performance", "UL", "thr_performance_UL.png"),
    ]:
        col = f"thr_kbps_{slice_name}_{link}"
        _line_plot(
            xs_by_name={n: m_eps[n]["step"] for n, *_ in methods},
            ys_by_name={n: m_eps[n][col]    for n, *_ in methods},
            xlabel="Episode Step",
            ylabel="Served Throughput (kbps)",
            title=f"Throughput — {slice_name.capitalize()} {link}",
            save_path=thr_dir / fn,
        )

    # Throughput violation rate: % of demand intervals where delivered < demanded
    for slice_name, link, fn in [
        ("critical",    "DL", "thr_violation_rate_critical_DL.png"),
        ("critical",    "UL", "thr_violation_rate_critical_UL.png"),
        ("performance", "DL", "thr_violation_rate_performance_DL.png"),
        ("performance", "UL", "thr_violation_rate_performance_UL.png"),
    ]:
        col = f"violation_rate_pct_{slice_name}_{link}"
        _line_plot(
            xs_by_name={n: m_eps[n]["step"] for n, *_ in methods},
            ys_by_name={n: m_eps[n][col]    for n, *_ in methods},
            xlabel="Episode Step",
            ylabel="Throughput Violation Rate (%)",
            title=f"Throughput Violation Rate — {slice_name.capitalize()} {link}",
            save_path=thr_dir / fn,
        )

    slice_link_labels = ["critical_DL", "critical_UL", "performance_DL", "performance_UL"]
    viol_col_map = {
        "critical_DL":    "violation_rate_pct_critical_DL",
        "critical_UL":    "violation_rate_pct_critical_UL",
        "performance_DL": "violation_rate_pct_performance_DL",
        "performance_UL": "violation_rate_pct_performance_UL",
    }
    gap_col_map = {
        "critical_DL":    "violation_gap_rate_critical_DL",
        "critical_UL":    "violation_gap_rate_critical_UL",
        "performance_DL": "violation_gap_rate_performance_DL",
        "performance_UL": "violation_gap_rate_performance_UL",
    }

    def _mean_ep_metric(ep_df, col_map):
        return {k: safe_mean(ep_df[v]) if v in ep_df.columns else 0.0 for k, v in col_map.items()}

    for mode_flag, tag, mode_label in mode_specs:
        mv, mg = {}, {}
        for name, *_ in methods:
            ep_m, _ = _subset_by_mode(m_eps[name], m_jobs[name], mode_flag)
            mv[name] = _mean_ep_metric(ep_m, viol_col_map)
            mg[name] = _mean_ep_metric(ep_m, gap_col_map)
        _bar_group_plot(
            slice_link_labels, mv,
            "Throughput Violation Rate (%)",
            f"Throughput Violation Rate — Slice/Link, {mode_label}",
            thr_dir / f"thr_violation_rate_{tag}.png",
        )
        _bar_group_plot(
            slice_link_labels, mg,
            "Throughput Violation Gap (normalised shortfall)",
            f"Throughput Violation Gap — Slice/Link, {mode_label}",
            thr_dir / f"thr_violation_gap_{tag}.png",
        )

    mv_all = {name: _mean_ep_metric(m_eps[name], viol_col_map) for name, *_ in methods}
    mg_all = {name: _mean_ep_metric(m_eps[name], gap_col_map)  for name, *_ in methods}
    _bar_group_plot(
        slice_link_labels, mv_all,
        "Throughput Violation Rate (%)",
        "Throughput Violation Rate — Slice/Link, Overall",
        thr_dir / "thr_violation_rate_overall.png",
    )
    _bar_group_plot(
        slice_link_labels, mg_all,
        "Throughput Violation Gap (normalised shortfall)",
        "Throughput Violation Gap — Slice/Link, Overall",
        thr_dir / "thr_violation_gap_overall.png",
    )

    # ════════════════════════════════════════════════════════════════════════
    # delay/
    # ════════════════════════════════════════════════════════════════════════

    for slice_name, link, fn in [
        ("critical",    "DL", "delay_critical_DL.png"),
        ("critical",    "UL", "delay_critical_UL.png"),
        ("performance", "DL", "delay_performance_DL.png"),
        ("performance", "UL", "delay_performance_UL.png"),
    ]:
        _line_plot(
            xs_by_name={n: episode_delay_timeseries(m_jobs[n], steps, slice_name, link)["step"]     for n, *_ in methods},
            ys_by_name={n: episode_delay_timeseries(m_jobs[n], steps, slice_name, link)["delay_ms"] for n, *_ in methods},
            xlabel="Episode Step",
            ylabel="Packet Delay (ms)",
            title=f"Packet Delay — {slice_name.capitalize()} {link}",
            save_path=delay_dir / fn,
        )

    def _mean_delay(job_df):
        return {
            "critical_DL":    safe_mean(job_df[(job_df["slice"] == "critical")    & (job_df["link"] == "DL")]["delay_ms"]),
            "critical_UL":    safe_mean(job_df[(job_df["slice"] == "critical")    & (job_df["link"] == "UL")]["delay_ms"]),
            "performance_DL": safe_mean(job_df[(job_df["slice"] == "performance") & (job_df["link"] == "DL")]["delay_ms"]),
            "performance_UL": safe_mean(job_df[(job_df["slice"] == "performance") & (job_df["link"] == "UL")]["delay_ms"]),
        }

    for mode_flag, tag, mode_label in mode_specs:
        md = {}
        for name, *_ in methods:
            ep_m, job_m = _subset_by_mode(m_eps[name], m_jobs[name], mode_flag)
            md[name] = _mean_delay(job_m)
        _bar_group_plot(
            slice_link_labels, md,
            "Average Packet Delay (ms)",
            f"Average Packet Delay — Slice/Link, {mode_label}",
            delay_dir / f"avg_delay_{tag}.png",
        )

    md_all = {name: _mean_delay(m_jobs[name]) for name, *_ in methods}
    _bar_group_plot(
        slice_link_labels, md_all,
        "Average Packet Delay (ms)",
        "Average Packet Delay — Slice/Link, Overall",
        delay_dir / "avg_delay_overall.png",
    )

    # ── Delay violation rate (deadline only) ──────────────────────────────
    # A job violates delay if it is not completed before its PDB deadline.
    # This metric stays deadline-based and is kept separate from QoS reliability.

    def _prepare_jobs_for_deadline_metrics(done_df, open_df):
        cols_base = ["app", "slice", "arrival_time_s", "done_time_s"]
        d = done_df.copy()
        for c in cols_base:
            if c not in d.columns:
                d[c] = np.nan
        d = d[cols_base].copy()
        if "dropped" in done_df.columns:
            drop_mask    = pd.to_numeric(done_df["dropped"], errors="coerce").fillna(0).astype(int).values
            done_time_ok = pd.to_numeric(done_df.get("done_time_s", np.nan), errors="coerce").notna().values
            d["done_flag"] = (drop_mask == 0) & done_time_ok
        else:
            d["done_flag"] = pd.to_numeric(d["done_time_s"], errors="coerce").notna()
        o = open_df.copy()
        for c in cols_base:
            if c not in o.columns:
                o[c] = np.nan
        o = o[cols_base].copy()
        o["done_flag"]   = False
        o["done_time_s"] = np.nan
        all_jobs = pd.concat([d, o], ignore_index=True)
        if all_jobs.empty:
            return all_jobs
        all_jobs["arrival_time_s"] = pd.to_numeric(all_jobs["arrival_time_s"], errors="coerce")
        all_jobs["done_time_s"]    = pd.to_numeric(all_jobs["done_time_s"],    errors="coerce")
        all_jobs = all_jobs.dropna(subset=["arrival_time_s"]).copy()
        all_jobs["arrival_step"]    = np.floor((all_jobs["arrival_time_s"] + 1e-12) / EPISODE_S).astype(int)
        all_jobs["pdb_ms"]          = all_jobs["app"].map(
            lambda a: float(APP_PACKET_PROFILE.get(str(a), {}).get("pdb_ms", np.nan))
        )
        all_jobs["deadline_time_s"] = all_jobs["arrival_time_s"] + (all_jobs["pdb_ms"] / 1000.0)
        return all_jobs

    def _mean_deadline_success_by_app(ep_df, done_df, open_df):
        job_df = _prepare_jobs_for_deadline_metrics(done_df, open_df)
        if job_df.empty:
            return {app: np.nan for app in APP_DEFS}
        sim_end = (int(pd.to_numeric(ep_df["step"], errors="coerce").max()) + 1) * EPISODE_S if not ep_df.empty else 0.0
        job_df  = job_df[job_df["deadline_time_s"] <= (sim_end + 1e-12)].copy()
        if job_df.empty:
            return {app: np.nan for app in APP_DEFS}
        job_df["deadline_success"] = (
            job_df["done_flag"].astype(bool)
            & job_df["done_time_s"].notna()
            & (job_df["done_time_s"] <= (job_df["deadline_time_s"] + 1e-12))
        )
        grp = job_df.groupby(["app", "arrival_step"], as_index=False).agg(
            total=("deadline_success", "size"), success=("deadline_success", "sum")
        )
        grp["deadline_success_pct"] = 100.0 * grp["success"] / grp["total"].clip(lower=1)
        out = {app: np.nan for app in APP_DEFS}
        for app in APP_DEFS:
            vals = grp[grp["app"] == app]["deadline_success_pct"]
            if len(vals):
                out[app] = float(np.nanmean(vals.values))
        return out

    def _filter_jobs_by_mode_arrival(ep_df, done_df, open_df, mode_flag):
        if "emergency_mode" in ep_df.columns:
            ep_sub = ep_df[ep_df["emergency_mode"] == mode_flag].copy()
        else:
            ep_sub = ep_df.copy() if mode_flag == 0 else ep_df.iloc[0:0].copy()
        if ep_sub.empty:
            return ep_sub, done_df.iloc[0:0].copy(), open_df.iloc[0:0].copy()
        step_set = set(pd.to_numeric(ep_sub["step"], errors="coerce").dropna().astype(int))
        if not step_set:
            return ep_sub, done_df.iloc[0:0].copy(), open_df.iloc[0:0].copy()
        def _filt(df):
            if df.empty:
                return df.copy()
            out = df.copy()
            arr_s = pd.to_numeric(out.get("arrival_time_s", np.nan), errors="coerce")
            out["_arr_step"] = np.floor((arr_s + 1e-12) / EPISODE_S)
            out = out.dropna(subset=["_arr_step"]).copy()
            out["_arr_step"] = out["_arr_step"].astype(int)
            return out[out["_arr_step"].isin(step_set)].drop(columns=["_arr_step"], errors="ignore")
        return ep_sub, _filt(done_df), _filt(open_df)

    def _fmt_dvr(v: float) -> str:
        if not np.isfinite(v):
            return "NA"
        if abs(v) <= 5e-12:
            return "0"
        if v < 0.01:
            return f"{v:.4f}"
        if v < 1.0:
            return f"{v:.3f}"
        return f"{v:.2f}"

    def _plot_delay_violation_bars(app_list, deadline_success_by_method, title, save_path):
        method_names = [m[0] for m in methods]
        x       = np.arange(len(app_list))
        w       = 0.18
        offsets = np.linspace(-1.5 * w, 1.5 * w, len(method_names))
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_REL)
        all_dvr_vals = []
        for name, off in zip(method_names, offsets):
            deadline_vals = np.array([deadline_success_by_method[name].get(app, np.nan) for app in app_list], dtype=float)
            dvr_vals = np.where(np.isfinite(deadline_vals), 100.0 - deadline_vals, np.nan)
            all_dvr_vals.extend(dvr_vals[np.isfinite(dvr_vals)].tolist())
            bars = ax.bar(x + off, dvr_vals, width=w, label=name)
            ax.bar_label(
                bars,
                labels=[_fmt_dvr(v) for v in dvr_vals],
                padding=4,
                fontsize=FONT_BAR_LABEL,
                rotation=90,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(app_list, rotation=0, fontsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ymax = max(all_dvr_vals) if all_dvr_vals else 1.0
        ax.set_ylim(0.0, ymax + max(2.0, 0.18 * ymax))
        ax.set_ylabel("Delay Violation Rate (%)", fontsize=FONT_LABEL)
        ax.legend(fontsize=FONT_LEGEND)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
        fig.tight_layout(pad=0.2)
        fig.savefig(save_path, dpi=PLOT_DPI, **SAVEFIG_KW)
        plt.close(fig)

    critical_apps    = [a for a, m in APP_DEFS.items() if m["slice"] == "critical" and a != "remote_engine_ctrl"]
    performance_apps = [a for a, m in APP_DEFS.items() if m["slice"] == "performance"]

    # Overall delay violation rate
    deadline_success_overall = {
        name: _mean_deadline_success_by_app(m_eps[name], m_jobs[name], m_open[name])
        for name, *_ in methods
    }
    _plot_delay_violation_bars(
        critical_apps, deadline_success_overall,
        "Delay Violation Rate — Critical Apps, Overall",
        delay_dir / "delay_violation_rate_critical_overall.png",
    )
    _plot_delay_violation_bars(
        performance_apps, deadline_success_overall,
        "Delay Violation Rate — Performance Apps, Overall",
        delay_dir / "delay_violation_rate_performance_overall.png",
    )

    # Normal / Emergency mode
    for mode_flag, tag, mode_label in mode_specs:
        deadline_success_mode = {}
        for name, *_ in methods:
            ep_sub, job_sub, open_sub = _filter_jobs_by_mode_arrival(m_eps[name], m_jobs[name], m_open[name], mode_flag)
            deadline_success_mode[name] = _mean_deadline_success_by_app(ep_sub, job_sub, open_sub)
        _plot_delay_violation_bars(
            critical_apps, deadline_success_mode,
            f"Delay Violation Rate — Critical Apps, {mode_label}",
            delay_dir / f"delay_violation_rate_critical_{tag}.png",
        )
        _plot_delay_violation_bars(
            performance_apps, deadline_success_mode,
            f"Delay Violation Rate — Performance Apps, {mode_label}",
            delay_dir / f"delay_violation_rate_performance_{tag}.png",
        )

    # ════════════════════════════════════════════════════════════════════════
    # reliability/
    # ════════════════════════════════════════════════════════════════════════

    def _mean_qos_reliability_by_app(ep_df, done_df, open_df):
        # In this simulator, a completed non-dropped job implies the requested
        # bits were fully delivered. QoS reliability therefore requires both:
        # 1) full job completion and 2) completion before the PDB deadline.
        job_df = _prepare_jobs_for_deadline_metrics(done_df, open_df)
        if job_df.empty:
            return {app: np.nan for app in APP_DEFS}
        sim_end = (int(pd.to_numeric(ep_df["step"], errors="coerce").max()) + 1) * EPISODE_S if not ep_df.empty else 0.0
        job_df  = job_df[job_df["deadline_time_s"] <= (sim_end + 1e-12)].copy()
        if job_df.empty:
            return {app: np.nan for app in APP_DEFS}

        job_df["qos_success"] = (
            job_df["done_flag"].astype(bool)
            & job_df["done_time_s"].notna()
            & (job_df["done_time_s"] <= (job_df["deadline_time_s"] + 1e-12))
        )

        out = {app: np.nan for app in APP_DEFS}
        for app in APP_DEFS:
            app_df = job_df[job_df["app"] == app]
            if len(app_df):
                out[app] = float(100.0 * app_df["qos_success"].sum() / max(len(app_df), 1))
        return out

    # Graduated y-axis (squeezes the 99–100% range so differences are visible)
    _REL_GRAD_VALS   = np.array([0., 20., 40., 60., 80., 90., 99., 99.9, 99.99, 99.999, 100.], dtype=float)
    _REL_GRAD_POS    = np.arange(len(_REL_GRAD_VALS), dtype=float)
    _REL_GRAD_LABELS = ["0", "20", "40", "60", "80", "90", "99", "99.9", "99.99", "99.999", "100"]

    def _to_rel_grad(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        ok  = np.isfinite(arr)
        if np.any(ok):
            out[ok] = np.interp(
                np.clip(arr[ok], _REL_GRAD_VALS[0], _REL_GRAD_VALS[-1]),
                _REL_GRAD_VALS, _REL_GRAD_POS,
            )
        return out

    def _fmt_rel(v: float) -> str:
        if not np.isfinite(v):
            return "NA"
        if abs(v) <= 5e-12:
            return "0"
        if abs(v - 100.0) <= 5e-4:
            return "100"
        if v < 99.0:
            return f"{v:.1f}"
        return f"{v:.4f}"

    def _plot_reliability_bars(app_list, rel_by_method, title, save_path):
        method_names = [m[0] for m in methods]
        x       = np.arange(len(app_list))
        w       = 0.18
        offsets = np.linspace(-1.5 * w, 1.5 * w, len(method_names))
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_REL)
        for name, off in zip(method_names, offsets):
            vals = np.array([rel_by_method[name].get(app, np.nan) for app in app_list], dtype=float)
            bars = ax.bar(x + off, _to_rel_grad(vals), width=w, label=name)
            ax.bar_label(
                bars,
                labels=[_fmt_rel(v) for v in vals],
                padding=4,
                fontsize=FONT_BAR_LABEL,
                rotation=90,
            )
        for i, app in enumerate(app_list):
            thr = float(APP_MIN_RELIABILITY_PCT.get(app, np.nan))
            if np.isfinite(thr):
                thr_g = _to_rel_grad(np.array([thr]))[0]
                if np.isfinite(thr_g):
                    ax.hlines(thr_g, i - 2.1*w, i + 2.1*w, colors="black", linestyles="--", linewidth=1.1)
                    ax.text(i + 2.22*w, thr_g, f"{thr:.5g}%", fontsize=12, va="bottom", ha="left")
        ax.set_ylim(-0.2, float(len(_REL_GRAD_POS) - 1) + 0.9)
        ax.set_yticks(_REL_GRAD_POS)
        ax.set_yticklabels(_REL_GRAD_LABELS)
        ax.set_xticks(x)
        ax.set_xticklabels(app_list, rotation=0, fontsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.set_ylabel("Reliability (%)", fontsize=FONT_LABEL)
        ax.legend(fontsize=FONT_LEGEND)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        fig.tight_layout(pad=0.2)
        fig.savefig(save_path, dpi=PLOT_DPI, **SAVEFIG_KW)
        plt.close(fig)

    # Overall reliability
    qos_rel_overall = {
        name: _mean_qos_reliability_by_app(m_eps[name], m_jobs[name], m_open[name])
        for name, *_ in methods
    }
    _plot_reliability_bars(
        critical_apps, qos_rel_overall,
        "Reliability — Critical Apps, Overall",
        reliability_dir / "reliability_critical_overall.png",
    )
    _plot_reliability_bars(
        performance_apps, qos_rel_overall,
        "Reliability — Performance Apps, Overall",
        reliability_dir / "reliability_performance_overall.png",
    )

    # Normal / Emergency mode
    for mode_flag, tag, mode_label in mode_specs:
        qos_rel_mode = {}
        for name, *_ in methods:
            ep_sub, job_sub, open_sub = _filter_jobs_by_mode_arrival(m_eps[name], m_jobs[name], m_open[name], mode_flag)
            qos_rel_mode[name] = _mean_qos_reliability_by_app(ep_sub, job_sub, open_sub)
        _plot_reliability_bars(
            critical_apps, qos_rel_mode,
            f"Reliability — Critical Apps, {mode_label}",
            reliability_dir / f"reliability_critical_{tag}.png",
        )
        _plot_reliability_bars(
            performance_apps, qos_rel_mode,
            f"Reliability — Performance Apps, {mode_label}",
            reliability_dir / f"reliability_performance_{tag}.png",
        )

    # Per-app QoS reliability progression over episode steps.
    def _qos_reliability_timeseries_by_app(ep_df, done_df, open_df):
        job_df = _prepare_jobs_for_deadline_metrics(done_df, open_df)
        if job_df.empty:
            return pd.DataFrame(columns=["app", "step", "reliability_pct"])
        sim_end = (int(pd.to_numeric(ep_df["step"], errors="coerce").max()) + 1) * EPISODE_S if not ep_df.empty else 0.0
        job_df  = job_df[job_df["deadline_time_s"] <= (sim_end + 1e-12)].copy()
        if job_df.empty:
            return pd.DataFrame(columns=["app", "step", "reliability_pct"])
        job_df["qos_success"] = (
            job_df["done_flag"].astype(bool)
            & job_df["done_time_s"].notna()
            & (job_df["done_time_s"] <= (job_df["deadline_time_s"] + 1e-12))
        )
        grp = job_df.groupby(["app", "arrival_step"], as_index=False).agg(
            total=("qos_success", "size"), success=("qos_success", "sum")
        )
        grp["reliability_pct"] = 100.0 * grp["success"] / grp["total"].clip(lower=1)
        return grp.rename(columns={"arrival_step": "step"})[["app", "step", "reliability_pct"]]

    rel_ts = {name: _qos_reliability_timeseries_by_app(m_eps[name], m_jobs[name], m_open[name]) for name, *_ in methods}

    for app in APP_DEFS:
        has_any = any(
            (not rel_ts[name].empty) and (app in rel_ts[name]["app"].values)
            for name, *_ in methods
        )
        if not has_any:
            continue
        plt.figure(figsize=PLOT_FIGSIZE_LINE)
        for name, *_ in methods:
            ts = rel_ts[name]
            if ts.empty:
                continue
            ts_app = ts[ts["app"] == app].sort_values("step")
            if ts_app.empty:
                continue
            plt.plot(ts_app["step"],
                     _to_rel_grad(ts_app["reliability_pct"].to_numpy(float)),
                     label=name, linewidth=2.0)
        thr = float(APP_MIN_RELIABILITY_PCT.get(app, np.nan))
        if np.isfinite(thr):
            thr_g = _to_rel_grad(np.array([thr]))[0]
            if np.isfinite(thr_g):
                plt.axhline(thr_g, color="black", linestyle="--", linewidth=1.2, label=f"target {thr:.5g}%")
        plt.xlabel("Episode Step", fontsize=FONT_LABEL)
        plt.ylabel("Reliability (%)", fontsize=FONT_LABEL)
        plt.ylim(-0.2, float(len(_REL_GRAD_POS) - 1) + 0.2)
        plt.yticks(_REL_GRAD_POS, _REL_GRAD_LABELS)
        plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
        plt.legend(fontsize=FONT_LEGEND)
        plt.tight_layout(pad=0.2)
        plt.savefig(reliability_dir / f"reliability_progression_{app}.png", dpi=PLOT_DPI, **SAVEFIG_KW)
        plt.close()

    # ════════════════════════════════════════════════════════════════════════
    # overhead/
    # ════════════════════════════════════════════════════════════════════════

    # PRB puncturing time-series
    plt.figure(figsize=PLOT_FIGSIZE_LINE)
    for name, *_ in methods:
        p = m_eps[name]["punctured_prbs_DL"] + m_eps[name]["punctured_prbs_UL"]
        plt.plot(m_eps[name]["step"], p, label=name)
    plt.xlabel("Episode Step", fontsize=FONT_LABEL)
    plt.ylabel("Punctured PRBs per Episode", fontsize=FONT_LABEL)
    plt.legend(fontsize=FONT_LEGEND)
    plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.tight_layout(pad=0.2)
    plt.savefig(overhead_dir / "puncturing_prbs_timeseries.png", dpi=PLOT_DPI, **SAVEFIG_KW)
    plt.close()

    # PRB puncturing distribution boxplot
    plt.figure(figsize=PLOT_FIGSIZE_BAR)
    box_data, box_labels = [], []
    for name, *_ in methods:
        p = (pd.to_numeric(m_eps[name]["punctured_prbs_DL"], errors="coerce").fillna(0.0)
           + pd.to_numeric(m_eps[name]["punctured_prbs_UL"], errors="coerce").fillna(0.0))
        box_data.append(p.values)
        box_labels.append(name)
    plt.boxplot(box_data, labels=box_labels, showfliers=True)
    plt.ylabel("Punctured PRBs per Episode", fontsize=FONT_LABEL)
    plt.xticks(rotation=10, fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.tight_layout(pad=0.2)
    plt.savefig(overhead_dir / "puncturing_prbs_boxplot.png", dpi=PLOT_DPI, **SAVEFIG_KW)
    plt.close()

    # Runtime, RAM, energy
    for col, ylabel, title, fn in [
        ("runtime_ms", "Scheduler Runtime (ms)", "Runtime",             "runtime_ms.png"),
        ("ram_mb",     "Memory Usage (MB)",       "Memory Usage",        "ram_mb.png"),
        ("energy_j",   "Energy Consumption (J)",  "Energy Consumption",  "energy_j.png"),
    ]:
        _line_plot(
            xs_by_name={n: m_eps[n]["step"] for n, *_ in methods},
            ys_by_name={n: m_eps[n][col]    for n, *_ in methods},
            xlabel="Episode Step",
            ylabel=ylabel,
            title=title,
            save_path=overhead_dir / fn,
        )

    # Energy efficiency: served kbits per joule, by slice/link
    eff_labels = ["Critical DL", "Performance DL", "Critical UL", "Performance UL"]
    eff_cols   = ["thr_kbps_critical_DL", "thr_kbps_performance_DL",
                  "thr_kbps_critical_UL", "thr_kbps_performance_UL"]
    x_eff = np.arange(len(eff_labels))
    plt.figure(figsize=PLOT_FIGSIZE_BAR)
    for name, *_ in methods:
        ep_cur = m_eps[name]
        e_j    = pd.to_numeric(ep_cur.get("energy_j", np.nan), errors="coerce")
        y_eff  = []
        for col in eff_cols:
            thr  = pd.to_numeric(ep_cur.get(col, np.nan), errors="coerce")
            kbit = thr * EPISODE_S
            eff  = np.where((e_j > 0) & np.isfinite(e_j), kbit / e_j, np.nan)
            y_eff.append(float(np.nanmean(eff)) if np.any(np.isfinite(eff)) else np.nan)
        plt.plot(x_eff, y_eff, marker="o", linewidth=2.4, markersize=7, label=name)
    plt.xticks(x_eff, eff_labels, fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.xlabel("Slice / Direction", fontsize=FONT_LABEL)
    plt.ylabel("Energy Efficiency (kbit / J)", fontsize=FONT_LABEL)
    plt.legend(fontsize=FONT_LEGEND)
    plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.tight_layout(pad=0.2)
    plt.savefig(overhead_dir / "energy_efficiency.png", dpi=PLOT_DPI, **SAVEFIG_KW)
    plt.close()
