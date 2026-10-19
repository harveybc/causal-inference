#!/usr/bin/env python3
"""Descriptive event-impact run over the two feature-eng test fixtures (calendar 2011-2021, EUR/USD hourly 2005-2020).

Everything this writes is DESCRIPTIVE / FIXTURE / CLOCK_UNVERIFIED / FIXTURE_RESEARCH_ONLY: the calendar fixture is an
Investing.com export whose terms forbid redistribution, so only aggregates and digests are written, never rows; the
hourly fixture's clock is inferred from the Nonfarm Payrolls spike, not read from a broker contract; and nothing
here is a confirmatory measurement. Rows at or after 2025-01-01 are refused by the loaders.

    CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 600s -n ci-event -- python3 provider/tools/event_impact_fixture_run.py \
        --calendar <feature-eng>/tests/data/economic_calendar_2011_2021.csv \
        --hourly <feature-eng>/tests/data/eurusd_hour_2005_2020_ohlc.csv --out docs/evidence/event_impact_20261019
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_inference_provider import event_impact as ei  # noqa: E402

LABELS = ["DESCRIPTIVE", "FIXTURE", "CLOCK_UNVERIFIED", "FIXTURE_RESEARCH_ONLY"]

FAMILIES = [
    "United States | Nonfarm Payrolls", "United States | CPI", "United States | Core CPI",
    "United States | Retail Sales", "United States | Core Retail Sales", "United States | ISM Manufacturing PMI",
    "United States | ISM Non-Manufacturing PMI", "United States | GDP", "United States | Unemployment Rate",
    "United States | ADP Nonfarm Employment Change", "United States | Fed Interest Rate Decision",
    "Euro Zone | ECB Interest Rate Decision", "Euro Zone | CPI", "Germany | German ZEW Economic Sentiment",
    "Germany | German Ifo Business Climate Index", "Germany | German Manufacturing PMI",
]

SHORT = {
    "United States | Nonfarm Payrolls": "NFP (EE.UU.)", "United States | CPI": "IPC (EE.UU.)",
    "United States | Core CPI": "IPC subyacente (EE.UU.)", "United States | Retail Sales": "Ventas minoristas (EE.UU.)",
    "United States | Core Retail Sales": "Ventas minoristas subyac. (EE.UU.)",
    "United States | ISM Manufacturing PMI": "ISM manufacturero", "United States | ISM Non-Manufacturing PMI": "ISM servicios",
    "United States | GDP": "PIB (EE.UU.)", "United States | Unemployment Rate": "Tasa de desempleo (EE.UU.)",
    "United States | ADP Nonfarm Employment Change": "ADP empleo", "United States | Fed Interest Rate Decision": "Tasa Fed",
    "Euro Zone | ECB Interest Rate Decision": "Tasa BCE", "Euro Zone | CPI": "IPC (zona euro)",
    "Germany | German ZEW Economic Sentiment": "ZEW (Alemania)", "Germany | German Ifo Business Climate Index": "Ifo (Alemania)",
    "Germany | German Manufacturing PMI": "PMI manufacturero (Alemania)",
}


def digest(path):
    h = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def broker_clock_scan(bars, nfp_utc):
    """The hourly fixture's clock, inferred: the offset (hours added to the calendar's UTC) at which the NFP bar's
    absolute return is largest against the same bar one week earlier and later."""
    abs_ret = bars["abs_ret"]
    out = {}
    for offset in range(-3, 6):
        values, controls = [], []
        for stamp in nfp_utc:
            bar = (stamp + pd.Timedelta(hours=offset)).floor("h")
            if bar in abs_ret.index:
                values.append(abs_ret[bar])
                for days in (-7, 7):
                    other = bar + pd.Timedelta(days=days)
                    if other in abs_ret.index:
                        controls.append(abs_ret[other])
        if values and controls:
            out[str(offset)] = {"n": len(values), "ratio": float(np.nanmean(values) / np.nanmean(controls))}
    best = max(out, key=lambda k: out[k]["ratio"])
    return {"ratios_by_offset_hours": out, "best_offset_hours": int(best),
            "reading": ("the hourly fixture is on a UTC clock (bars labelled by open time) if the best offset is 0; "
                        "CLOCK_UNVERIFIED: inferred from the spike, not read from a broker contract")}


def poc_comparison(bars, nfp_utc, ours):
    """What the earlier NFP POC measured (first Friday, broker hour 15, vs other Fridays at 15) next to ours."""
    index = bars.index
    fridays = index[(index.dayofweek == 4) & (index.hour == 15)]
    first_fridays = {d for d in pd.DatetimeIndex(fridays).normalize() if d.day <= 7}
    nfp_days = set(pd.DatetimeIndex(nfp_utc).normalize())
    poc_event = [bars.at[t, "abs_ret"] for t in fridays if t.normalize() in first_fridays]
    poc_control = [bars.at[t, "abs_ret"] for t in fridays if t.normalize() not in nfp_days]
    lag0 = ours["lags"].index(0)
    return {
        "poc_recorded": {"vol_ratio": 1.07, "drift": "not significant at 1/2/4/8h", "clock": "UTC+2 (EET), bar 15",
                         "events": "first Friday of each month, no calendar"},
        "poc_measure_replayed_here": {"ratio_bar15_first_fridays_vs_other_fridays":
                                      float(np.nanmean(poc_event) / np.nanmean(poc_control)), "n": len(poc_event)},
        "this_run": {"ratio_release_bar_vs_matched_controls": ours["ratio"][lag0],
                     "excess_ci": [ours["excess_ci_lower"][lag0], ours["excess_ci_upper"][lag0]],
                     "n_events": ours["n_events"]},
        "explanation": ("The POC scanned broker offsets by comparing first-Friday bars to the previous Thursday at the "
                        "same hour and picked hour 15 (it read 'UTC+2'); the fixture is on a UTC clock and NFP is "
                        "published at 13:30 UTC in winter and 12:30 UTC in summer, so bar 15 is two to three hours "
                        "AFTER the release, where the excess has mostly dissipated (half-life below two hours). It "
                        "also used first Fridays instead of the calendar (NFP moves to Thursday or the second "
                        "Friday around holidays) and one-bar open-to-close moves. At the release bar, against "
                        "matched same-hour same-weekday controls, the ratio is above two. The drift verdict does not "
                        "change: the unconditional drift is ~0 here too; the trend exists only through the surprise."),
    }


def run(calendar_path, hourly_path, out_dir, *, start="2011-01-01", end="2021-01-01", decision_start="2018-01-01",
        window=24, n_boot=400, seed=0, families=FAMILIES):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calendar = ei.load_calendar(calendar_path, end_exclusive=end)
    calendar = calendar[calendar["release_utc"] >= pd.Timestamp(start)]
    bars = ei.load_hourly(hourly_path, end_exclusive=end)
    bars = bars[bars.index >= pd.Timestamp(start) - pd.Timedelta(days=30)]
    raw = pd.read_csv(hourly_path)
    stamps = pd.to_datetime(raw["datetime"], dayfirst=True)
    raw = raw.assign(datetime=stamps).sort_values("datetime").set_index("datetime")
    raw = raw[(raw.index >= bars.index[0]) & (raw.index < pd.Timestamp(end))]
    bars["high"], bars["low"] = raw["high"].reindex(bars.index), raw["low"].reindex(bars.index)

    nfp = calendar[calendar["family"] == "United States | Nonfarm Payrolls"]
    clock = {
        "calendar": {
            "clock": "fixed UTC-5 without daylight saving for 2011-01 to 2018-01, exporter pseudo-DST afterwards; "
                     "offset per row from 08:30-Eastern anchor releases (never a legal DST rule)",
            "anchor_families": list(ei.ANCHOR_FAMILIES_0830_ET),
            "rows_by_status": calendar["clock_status"].value_counts().to_dict(),
            "months_with_two_offsets": sorted(str(m) for m, g in calendar.groupby(calendar["release_local"].dt.to_period("M"))
                                              if g["utc_offset_hours"].nunique() > 1),
            "nfp_release_utc_hours": nfp["release_utc"].dt.strftime("%H:%M").value_counts().to_dict(),
            "mapping_check_2011_01_07": nfp[nfp["release_local"].dt.date == pd.Timestamp("2011-01-07").date()]
            [["release_local", "release_utc", "actual", "forecast"]].astype(str).to_dict("records"),
        },
        "hourly": broker_clock_scan(bars, nfp["release_utc"]),
    }

    results, half_life_rows, figures_data = {}, [], {}
    # control windows must be free of any HIGH-impact release of the three economies, not of every low-impact row
    central_banks = ["United States | FOMC Statement", "United States | Fed Interest Rate Decision",
                     "United States | FOMC Press Conference", "Euro Zone | ECB Interest Rate Decision",
                     "Euro Zone | ECB Press Conference", "United States | PPI", "United States | Initial Jobless Claims"]
    exclusion = calendar[calendar["family"].isin(list(families) + central_banks)]
    for family in families:
        try:
            est = ei.estimate_family(bars, calendar, family, window=window, n_boot=n_boot, seed=seed,
                                     exclusion_calendar=exclusion)
        except ei.EventImpactError as error:
            results[family] = {"status": "REFUSED", "code": error.code, "why": error.why}
            continue
        rows = calendar[calendar["family"] == family].sort_values("release_utc")
        positions = ei.release_positions(bars, rows["release_utc"])
        rows, positions = rows[positions >= 0], positions[positions >= 0]
        z, _ = ei.standardize_surprises(rows["surprise"].to_numpy())
        try:
            trend = ei.trend_impulse_response(bars, positions, z, window=12, n_boot=n_boot, seed=seed)
            comparison = ei.compare_half_lives(est["impulse_response"], trend)
        except ei.EventImpactError as error:
            trend, comparison = {"status": "REFUSED", "code": error.code, "why": error.why}, {"status": "NOT_COMPARABLE"}
        state = ei.market_state(bars, positions)
        times = pd.DatetimeIndex(rows["release_utc"])
        fit_mask = np.asarray(times < pd.Timestamp(decision_start))
        y4 = ei.forward_returns(bars, positions, 4)
        arms = {}
        arms["regime_unsupervised"] = ei.cate_out_of_sample(
            y4, z, ei.regime_labels(state, fit_mask), times, decision_start=decision_start)
        arms["fs_rl_ohlc_subset"] = ei.cate_out_of_sample(
            y4, z, ei.regime_labels(state, fit_mask, columns=("px.rv5", "rg.atr_ratio")), times,
            decision_start=decision_start)
        arms["fs_rl_declared"] = {
            "feature_manifest_sha256": ei.FS_RL_FEATURE_MANIFEST_SHA256, "features": list(ei.FS_RL_FEATURES),
            "available_in_fixture": list(ei.FS_RL_OHLC_SUBSET),
            "status": "PARTIAL_OHLC_SUBSET_ONLY",
            "note": "selected for price regression (I6-DMO, 2024 development), not for effect estimation",
        }
        results[family] = {**est, "trend_impulse_response": trend, "half_life_comparison": comparison,
                           "state_arms": arms}
        ir = est["impulse_response"]
        half_life_rows.append({
            "family": family, "n_events": ir["n_events"], "vol_ratio_release_bar": ir["ratio"][ir["lags"].index(0)],
            "clock": est["clock_check"].get("verdict"), "clock_best_offset_h": est["clock_check"].get("best_offset_hours"),
            "vol_half_life_h": ir["half_life_hours"], "vol_half_life_ci": ir["half_life_ci"],
            "vol_half_life_censored": ir["half_life_censored"],
            "trend_half_life_censored": trend.get("half_life_censored"),
            "vol_fit_r2": ir["half_life_fit_r2"],
            "trend_half_life_h": trend.get("half_life_hours"), "trend_half_life_ci": trend.get("half_life_ci"),
            "half_lives": comparison.get("verdict", comparison.get("status")),
            "drift_verdict": est["surprise_drift"]["verdict"],
            "drift_slope_4h": est["surprise_drift"]["by_horizon"]["4"].get("slope"),
            "drift_slope_4h_p": est["surprise_drift"]["by_horizon"]["4"].get("slope_p"),
        })
        figures_data[family] = {"ir": ir, "trend": trend, "z": z, "y4": y4, "drift": est["surprise_drift"]}

    fitted = [f for f in families if results[f].get("impulse_response")]
    concurrency = ei.concurrency_test(bars, calendar, fitted, window=window)
    selection = ei.select_events(bars, calendar, fitted, decision_start=decision_start, horizon=4, window=window,
                                 n_boot=min(n_boot, 200), seed=seed, exclusion_calendar=exclusion)
    nfp_ir = results.get("United States | Nonfarm Payrolls", {}).get("impulse_response")
    poc = poc_comparison(bars, nfp["release_utc"], nfp_ir) if nfp_ir else {"status": "NFP_NOT_FITTED"}

    document = ei.strip_private({
        "schema": ei.SCHEMA + ".fixture_run", "labels": LABELS, "generated_utc": ei.utc_now_iso(),
        "inputs": {"calendar_sha256": digest(calendar_path), "hourly_sha256": digest(hourly_path),
                   "calendar_rows_read": int(len(calendar)), "bars_read": int(len(bars)),
                   "period": [start, end], "decision_start": decision_start,
                   "test_boundary_never_read": str(ei.TEST_BOUNDARY.date()),
                   "licensing": "Investing.com export: RESEARCH ONLY, no rows are written here"},
        "parameters": {"window_hours": window, "pre_window_hours": 24, "n_controls": 5, "n_boot": n_boot,
                       "control_exclusion": "every covered family plus FOMC/Fed/ECB decisions and statements, PPI "
                                            "and Initial Jobless Claims, inside [c - window, c + window]",
                       "control_exclusion_rows": int(len(exclusion)),
                       "seed": seed, "horizons": [1, 2, 4, 8], "min_prior_releases_for_scale": ei.MIN_PRIOR_RELEASES},
        "clock": clock, "families": results, "half_life_table": half_life_rows,
        "concurrency": concurrency, "event_selection": selection, "nfp_poc_comparison": poc,
        "identification": {
            "verdict": "DESCRIPTIVE_MATCHED_CONTRAST_NOT_CONFIRMATORY",
            "reasons": ["the hourly fixture's clock is inferred (CLOCK_UNVERIFIED)",
                        "the calendar's publication instant is the scheduled one; no receipt timestamp exists",
                        "matching removes hour-of-day, weekday and pre-volatility confounding only; concurrent "
                        "unscheduled news is not observed",
                        "the fixture is research-only data"],
        },
    })
    (out_dir / "event_impact_fixture_run.json").write_text(json.dumps(document, indent=1, sort_keys=True),
                                                           encoding="utf-8")
    write_tables(out_dir, document)
    make_figures(out_dir, figures_data, half_life_rows)
    selection_manifest = {
        "schema": "causal-inference.selected_events.v1", "labels": LABELS, "frozen": True,
        "inputs": document["inputs"], "decision_start": decision_start,
        "selected": selection["selected"], "ranking": selection["ranking"],
        "run_sha256": sha256((out_dir / "event_impact_fixture_run.json").read_bytes()).hexdigest(),
    }
    (out_dir / "SELECTED_EVENTS_MANIFEST.json").write_text(json.dumps(selection_manifest, indent=1, sort_keys=True),
                                                            encoding="utf-8")
    return document


def write_tables(out_dir, document):
    lines = ["# Tabla de vidas medias (DESCRIPTIVE / FIXTURE / CLOCK_UNVERIFIED / FIXTURE_RESEARCH_ONLY)", "",
             "| familia | n | ratio vol. barra de publicación | reloj | vida media vol. (h) [IC 95%] | vida media tendencia (h) [IC 95%] | ¿difieren? | veredicto deriva | pendiente 4h por z | p |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    def name(family):
        return family.replace(" | ", " — ")

    for r in document["half_life_table"]:
        def fmt(v):
            return "n/d" if v is None else f"{v:.2f}"

        def ci(c):
            return "n/d" if not c or c[0] is None else f"[{c[0]:.2f}, {c[1]:.2f}]"
        lines.append(f"| {name(r['family'])} | {r['n_events']} | {fmt(r['vol_ratio_release_bar'])} | {r['clock']} | "
                     f"{fmt(r['vol_half_life_h'])} {ci(r['vol_half_life_ci'])}"
                     f"{' (censurada)' if r['vol_half_life_censored'] else ''} | {fmt(r['trend_half_life_h'])} "
                     f"{ci(r['trend_half_life_ci'])} | {r['half_lives']} | {r['drift_verdict']} | "
                     f"{'n/d' if r['drift_slope_4h'] is None else format(r['drift_slope_4h'], '.2e')} | "
                     f"{fmt(r['drift_slope_4h_p'])} |")
    lines += ["", "## Selección de eventos (periodo de decisión, FDR Benjamini-Hochberg)", "",
              "| familia | estado | n entren. | n decisión | dirección | media OOS (log) | efecto/riesgo OOS | p OOS | BH | seleccionada |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for e in document["event_selection"]["ranking"]:
        if e["status"] != "OK":
            lines.append(f"| {name(e['family'])} | {e['status']} | {e['n_train']} | {e['n_decision']} | | | | | | |")
            continue
        lines.append(f"| {name(e['family'])} | OK | {e['n_train']} | {e['n_decision']} | {e['trade_direction']:+.0f} | "
                     f"{e['oos_mean_log_return']:.2e} | {e['oos_effect_per_unit_risk']:.3f} | {e['oos_p']:.3f} | "
                     f"{e['bh_pass']} | {e['selected']} |")
    (out_dir / "HALF_LIFE_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(out_dir, data, half_life_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = list(data)
    if not families:
        return
    cols = 4
    rows = int(np.ceil(len(families) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, family in zip(axes.ravel(), families):
        ir = data[family]["ir"]
        lags = np.array(ir["lags"])
        ax.fill_between(lags, np.array(ir["excess_ci_lower"]) * 1e4, np.array(ir["excess_ci_upper"]) * 1e4,
                        color="#4c72b0", alpha=0.2, label="IC 95% exceso")
        ax.plot(lags, np.array(ir["excess"]) * 1e4, color="#4c72b0", lw=2, label="exceso (evento − control)")
        ax.plot(lags, np.array(ir["event_mean"]) * 1e4, color="#c44e52", lw=1, ls="--", label="evento")
        ax.plot(lags, np.array(ir["control_mean"]) * 1e4, color="#55a868", lw=1, ls=":", label="controles emparejados")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        hl = ir["half_life_hours"]
        ax.set_title(f"{SHORT.get(family, family)} (n={ir['n_events']}, t½={hl:.1f} h)", fontsize=9)
        ax.set_xlabel("horas desde la publicación")
        ax.set_ylabel("|retorno| medio (pb)")
    for ax in axes.ravel()[len(families):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Respuesta al impulso de la volatilidad realizada por familia de evento — EUR/USD 1h, 2011–2020\n"
                 "DESCRIPTIVO · FIXTURE · RELOJ NO VERIFICADO · SOLO INVESTIGACIÓN", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"impulso_volatilidad_por_familia.{ext}", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, family in zip(axes.ravel(), families):
        z, y4, drift = data[family]["z"], data[family]["y4"], data[family]["drift"]
        ok = np.isfinite(z) & np.isfinite(y4)
        ax.scatter(z[ok], y4[ok] * 1e4, s=10, alpha=0.6, color="#4c72b0")
        fit = drift["by_horizon"]["4"]
        if fit.get("status") == "OK":
            grid = np.linspace(np.nanmin(z[ok]), np.nanmax(z[ok]), 50)
            ax.plot(grid, (fit["intercept"] + fit["slope"] * grid) * 1e4, color="#c44e52", lw=1.5,
                    label=f"pendiente {fit['slope']*1e4:.1f} pb/z, p={fit['slope_p']:.2f}")
            ax.legend(fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{SHORT.get(family, family)} — {drift['verdict']}", fontsize=8)
        ax.set_xlabel("sorpresa estandarizada z (real − consenso) / σ previa")
        ax.set_ylabel("retorno a 4 h (pb)")
    for ax in axes.ravel()[len(families):]:
        ax.axis("off")
    fig.suptitle("Sorpresa frente a deriva (retorno EUR/USD a 4 h tras la publicación) por familia\n"
                 "DESCRIPTIVO · FIXTURE · RELOJ NO VERIFICADO · SOLO INVESTIGACIÓN", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sorpresa_vs_deriva_por_familia.{ext}", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, family in zip(axes.ravel(), families):
        tr = data[family]["trend"]
        if "lags" not in tr:
            ax.set_title(f"{SHORT.get(family, family)}: {tr.get('code', 'n/d')}", fontsize=8)
            continue
        lags = np.array(tr["lags"])
        ax.fill_between(lags, np.array(tr["beta_ci_lower"]) * 1e4, np.array(tr["beta_ci_upper"]) * 1e4,
                        color="#dd8452", alpha=0.25)
        ax.plot(lags, np.array(tr["beta"]) * 1e4, color="#dd8452", lw=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(f"{SHORT.get(family, family)} (t½ tendencia={tr['half_life_hours']:.1f} h)", fontsize=9)
        ax.set_xlabel("horas desde la publicación")
        ax.set_ylabel("β por unidad de z (pb)")
    for ax in axes.ravel()[len(families):]:
        ax.axis("off")
    fig.suptitle("Respuesta al impulso de la tendencia (deriva firmada por unidad de sorpresa) por familia\n"
                 "DESCRIPTIVO · FIXTURE · RELOJ NO VERIFICADO · SOLO INVESTIGACIÓN", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"impulso_tendencia_por_familia.{ext}", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 0.45 * len(half_life_rows) + 1.5))
    ax.axis("off")
    cells = [[SHORT.get(r["family"], r["family"]), r["n_events"], f"{r['vol_ratio_release_bar']:.2f}",
              f"{r['vol_half_life_h']:.2f}",
              (f"[{r['vol_half_life_ci'][0]:.2f}, {r['vol_half_life_ci'][1]:.2f}]" if r["vol_half_life_ci"] and
               r["vol_half_life_ci"][0] is not None and np.isfinite(r["vol_half_life_ci"][0]) else "n/d"),
              (f"{r['trend_half_life_h']:.2f}" if r["trend_half_life_h"] is not None and
               np.isfinite(r["trend_half_life_h"]) else "n/d"),
              r["half_lives"], r["drift_verdict"]] for r in half_life_rows]
    table = ax.table(cellText=cells, colLabels=["familia", "n", "ratio vol. barra", "t½ vol. (h)", "IC 95% t½ vol.",
                                                "t½ tendencia (h)", "¿difieren?", "veredicto deriva"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.3)
    ax.set_title("Tabla de vidas medias por familia — DESCRIPTIVO · FIXTURE · RELOJ NO VERIFICADO · SOLO INVESTIGACIÓN",
                 fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"tabla_vidas_medias.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--hourly", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2021-01-01")
    parser.add_argument("--decision-start", default="2018-01-01")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--n-boot", type=int, default=400)
    args = parser.parse_args(argv)
    document = run(args.calendar, args.hourly, args.out, start=args.start, end=args.end,
                   decision_start=args.decision_start, window=args.window, n_boot=args.n_boot)
    for row in document["half_life_table"]:
        print(f"{row['family']:<50} n={row['n_events']:>4} ratio={row['vol_ratio_release_bar']:.2f} "
              f"t1/2 vol={row['vol_half_life_h']:.2f}h[{row['vol_half_life_censored']}] trend={row['trend_half_life_h']} "
              f"drift={row['drift_verdict']} clock={row['clock']}({row['clock_best_offset_h']})")
    print("selected:", document["event_selection"]["selected"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
