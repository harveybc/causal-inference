"""Episode construction for PS3-C: economic events (and indicator/regime crossings) -> EURUSD.

One row = one historical episode anchored at the decision instant t. Treatment A is built
before any outcome is read; outcomes are realised after t; the history H / covariates W use
only bars that CLOSED at or before t. Everything is cut at ``train_end`` first: bars and
events after it are dropped before any statistic (scale, threshold, regime cut) is fitted,
so perturbing them cannot change a single output (FS01 style, tested).

Clocks: ``published_at`` (source clock) and ``received_at`` (our clock) are kept apart;
the decision instant is the later of the two. An event without an availability instant is
excluded and counted (``MISSING_AVAILABILITY``), never imputed.

Outcomes use elapsed time, not rows: the price at t+h is the last close at or before t+h
(weekends are elapsed hours). An outcome whose window crosses the last TRAIN bar is NaN and
counted (``OUTCOME_WINDOW_CROSSES_TRAIN_END``).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SHORT_H = (1, 2, 3, 4, 5, 6)
LONG_H = (24, 48, 72, 96, 120, 144)
PLACEBO_H = (1, 6, 24)
HOUR = np.int64(3600 * 10**9)


class EpisodeError(ValueError):
    """Input violates a point-in-time precondition (strict mode)."""


def _utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


@dataclass
class EpisodeSet:
    episodes: pd.DataFrame
    pseudo: pd.DataFrame
    exclusions: Counter
    manifest: dict
    fitted: dict = field(default_factory=dict)


class PriceGrid:
    """Bars indexed by close time (UTC ns), cut at ``train_end`` before anything else."""

    def __init__(self, bars, *, train_end, bar_minutes=60, bar_label="open", time_col="timestamp"):
        b = bars.copy()
        raw = b[time_col] if time_col in b.columns else pd.Series(b.index, index=b.index)
        ts = _utc(raw)
        if bar_label == "open":
            ts = ts + pd.Timedelta(minutes=bar_minutes)
        b = b.assign(_close_time=ts).dropna(subset=["_close_time", "close"])
        cut = _utc(pd.Series([train_end])).iloc[0]
        self.train_end = cut
        b = b[b["_close_time"] <= cut].sort_values("_close_time")
        b = b[~b["_close_time"].duplicated(keep="last")]
        if b.empty:
            raise EpisodeError("no TRAIN bars at or before train_end")
        self.t = b["_close_time"].values.astype("datetime64[ns]").astype(np.int64)
        self.logc = np.log(b["close"].to_numpy(dtype=float))
        self.close = b["close"].to_numpy(dtype=float)
        self.high = b["high"].to_numpy(dtype=float) if "high" in b else None
        self.low = b["low"].to_numpy(dtype=float) if "low" in b else None
        self.bar = np.int64(bar_minutes * 60 * 10**9)
        self.last = self.t[-1]
        self.first = self.t[0]
        self.digest = hashlib.sha256(np.ascontiguousarray(self.t).tobytes() + np.ascontiguousarray(self.close).tobytes()).hexdigest()

    def idx_le(self, tau):
        """Index of the last bar that closed at or before tau (or -1)."""
        return int(np.searchsorted(self.t, tau, side="right")) - 1

    def idx_ge(self, tau):
        i = int(np.searchsorted(self.t, tau, side="left"))
        return i if i < len(self.t) else -1

    def logp_le(self, tau, max_stale=None):
        if tau > self.last or tau < self.first:
            return np.nan
        i = self.idx_le(tau)
        if i < 0:
            return np.nan
        if max_stale is not None and tau - self.t[i] > max_stale:
            return np.nan
        return self.logc[i]

    def returns_window(self, end_tau, hours):
        """Hourly log returns of bars closing in (end_tau - hours, end_tau]."""
        i1 = self.idx_le(end_tau)
        i0 = int(np.searchsorted(self.t, end_tau - hours * HOUR, side="right"))
        if i1 < 1 or i0 > i1:
            return np.array([])
        i0 = max(i0, 1)
        return np.diff(self.logc[i0 - 1:i1 + 1])


def _price_block(g, anchor, entry, short_h, long_h, placebo_h, history_h, barrier, ex):
    """W (pre-anchor), Y (post-entry), Y_pre (placebo) for one anchor. NaN with counted reasons."""
    out = {}
    lp_a = g.logp_le(anchor, max_stale=72 * HOUR)
    if not np.isfinite(lp_a):
        ex["INSUFFICIENT_HISTORY"] += 1
        return None
    for h in history_h:
        past = g.logp_le(anchor - h * HOUR)
        out[f"W_ret_{h}h"] = lp_a - past if np.isfinite(past) else np.nan
        r = g.returns_window(anchor, h)
        out[f"W_rv_{h}h"] = float(np.sqrt(np.sum(r * r))) if len(r) >= max(2, h // 4) else np.nan
    for h in placebo_h:
        past = g.logp_le(anchor - h * HOUR)
        out[f"Ypre_{h}h"] = lp_a - past if np.isfinite(past) else np.nan
    ie = g.idx_ge(entry)
    if ie < 0 or g.t[ie] - entry > 2 * g.bar:
        ex["NO_ENTRY_BAR"] += 1
        return None
    e_t, lp_e = g.t[ie], g.logc[ie]
    out["entry_time"] = pd.Timestamp(e_t, tz="UTC")
    for name, hs in (("Y_s", short_h), ("Y_l", long_h)):
        for h in hs:
            tgt = e_t + h * HOUR
            if tgt > g.last:
                ex[f"OUTCOME_WINDOW_CROSSES_TRAIN_END:{name}_{h}h"] += 1
                out[f"{name}_{h}h"] = np.nan
            else:
                out[f"{name}_{h}h"] = g.logp_le(tgt) - lp_e
    out["outcome_last_instant"] = pd.Timestamp(min(e_t + max(long_h or short_h) * HOUR, g.last), tz="UTC")
    if barrier:
        out["Y_b"] = _barrier(g, ie, anchor, barrier, ex)
    return out


def _barrier(g, ie, anchor, barrier, ex):
    """+1 TP first, -1 SL first, 0 timeout; NaN ambiguous/censored (counted). Long side, close-only entry."""
    ia = g.idx_le(anchor)
    lo_i = int(np.searchsorted(g.t, anchor - barrier.get("atr_h", 24) * HOUR, side="right"))
    if g.high is not None and g.low is not None:
        atr = float(np.nanmean(g.high[lo_i:ia + 1] - g.low[lo_i:ia + 1])) if ia >= lo_i else np.nan
    else:
        atr = float(np.nanmean(np.abs(np.diff(g.close[max(lo_i - 1, 0):ia + 1])))) if ia > lo_i else np.nan
    if not np.isfinite(atr) or atr <= 0:
        ex["Y_b:ATR_UNDEFINED"] += 1
        return np.nan
    entry = g.close[ie]
    tp, sl = entry + barrier.get("tp_atr", 2.0) * atr, entry - barrier.get("sl_atr", 2.0) * atr
    end_t = g.t[ie] + barrier.get("timeout_h", 120) * HOUR
    if end_t > g.last:
        ex["Y_b:CENSORED_AT_TRAIN_END"] += 1
        return np.nan
    j1 = g.idx_le(end_t)
    hi = g.high[ie + 1:j1 + 1] if g.high is not None else g.close[ie + 1:j1 + 1]
    lo = g.low[ie + 1:j1 + 1] if g.low is not None else g.close[ie + 1:j1 + 1]
    up, dn = hi >= tp, lo <= sl
    k_up = int(np.argmax(up)) if up.any() else None
    k_dn = int(np.argmax(dn)) if dn.any() else None
    if k_up is None and k_dn is None:
        return 0.0
    if k_up is not None and k_dn is not None and k_up == k_dn:
        ex["Y_b:AMBIGUOUS_INTRABAR"] += 1
        return np.nan
    if k_dn is None or (k_up is not None and k_up < k_dn):
        return 1.0
    return -1.0


def _calendar(ts):
    h = ts.hour + ts.minute / 60.0
    d = ts.dayofweek
    return {"W_hour_sin": np.sin(2 * np.pi * h / 24), "W_hour_cos": np.cos(2 * np.pi * h / 24),
            "W_dow_sin": np.sin(2 * np.pi * d / 7), "W_dow_cos": np.cos(2 * np.pi * d / 7)}


def _regimes(df, col, cuts=None):
    v = df[col]
    if cuts is None:
        cuts = [float(x) for x in np.nanquantile(v, [1 / 3, 2 / 3])] if v.notna().any() else [np.nan, np.nan]
    lab = np.where(v.isna(), None, np.where(v <= cuts[0], "low_vol", np.where(v <= cuts[1], "mid_vol", "high_vol")))
    return lab, cuts


def build_event_episodes(events, bars, *, train_end, asset="EURUSD", bar_minutes=60, bar_label="open",
                         short_h=SHORT_H, long_h=LONG_H, placebo_h=PLACEBO_H, history_h=(24, 120),
                         barrier=None, expectation="consensus", scale_min_n=10, pseudo_offset_h=168,
                         pseudo_exclusion_h=12, strict=False, time_col="timestamp"):
    """Episodes of economic releases with A = (actual_initial - prior expectation) / TRAIN scale of the type.

    ``events`` columns: event_type, published_at, actual; optional received_at, consensus,
    previous, currency, event_id. ``expectation``: "consensus" (published consensus only) or
    "consensus_or_previous" (falls back to the previous value, labelled MODEL_BASED_EXPECTATION).
    A = 0 means "published equal to the expectation", never "no release".
    """
    barrier = {"tp_atr": 2.0, "sl_atr": 2.0, "timeout_h": 120, "atr_h": 24} if barrier is None else barrier
    g = PriceGrid(bars, train_end=train_end, bar_minutes=bar_minutes, bar_label=bar_label, time_col=time_col)
    ex = Counter()
    ev = events.copy()
    for c in ("event_type", "published_at", "actual"):
        if c not in ev.columns:
            raise EpisodeError(f"events missing column {c}")
    ev["published_at"] = _utc(ev["published_at"])
    ev["received_at"] = (_utc(ev["received_at"]) if "received_at" in ev.columns
                         else pd.Series(pd.NaT, index=ev.index, dtype="datetime64[ns, UTC]"))
    miss = ev["published_at"].isna()
    if miss.any():
        if strict:
            raise EpisodeError(f"MISSING_AVAILABILITY: {int(miss.sum())} events without published_at")
        ex["MISSING_AVAILABILITY"] += int(miss.sum())
        ev = ev[~miss]
    bad = ev["received_at"].notna() & (ev["received_at"] < ev["published_at"])
    if bad.any():
        if strict:
            raise EpisodeError(f"RECEIVED_BEFORE_PUBLISHED: {int(bad.sum())} events")
        ex["RECEIVED_BEFORE_PUBLISHED"] += int(bad.sum())
        ev = ev[~bad]
    ev["decision_time"] = ev[["published_at", "received_at"]].max(axis=1)
    beyond = ev["decision_time"] > g.train_end
    ex["BEYOND_TRAIN_END_NOT_READ"] += int(beyond.sum())
    ev = ev[~beyond].sort_values("decision_time").reset_index(drop=True)
    # expectation and raw surprise
    cons = pd.to_numeric(ev["consensus"], errors="coerce") if "consensus" in ev.columns else pd.Series(np.nan, index=ev.index)
    prev = pd.to_numeric(ev["previous"], errors="coerce") if "previous" in ev.columns else pd.Series(np.nan, index=ev.index)
    act = pd.to_numeric(ev["actual"], errors="coerce")
    kind = np.where(cons.notna(), "PUBLISHED_CONSENSUS",
                    np.where((expectation == "consensus_or_previous") & prev.notna(), "MODEL_BASED_EXPECTATION", "NONE"))
    expect = np.where(cons.notna(), cons, np.where(kind == "MODEL_BASED_EXPECTATION", prev, np.nan))
    ev["expectation_kind"] = kind
    ev["raw_surprise"] = act - expect
    ex["NO_EXPECTATION"] += int((kind == "NONE").sum())
    ex["NO_ACTUAL"] += int(act.isna().sum())
    # TRAIN-fitted robust scale per event type (fitted on TRAIN rows only: everything here is <= train_end)
    scales = {}
    for et, grp in ev.groupby("event_type"):
        r = grp["raw_surprise"].dropna()
        if len(r) < scale_min_n:
            scales[et] = None
            continue
        mad = float(np.median(np.abs(r - np.median(r)))) * 1.4826
        sd = float(r.std(ddof=1))
        scales[et] = mad if mad > 0 else (sd if sd > 0 else None)
    sc = ev["event_type"].map(lambda e: scales.get(e))
    ev["A_surprise"] = np.where(sc.notna(), ev["raw_surprise"] / sc.astype(float), np.nan)
    ex["SCALE_UNDEFINED"] += int((sc.isna() & ev["raw_surprise"].notna()).sum())
    # build rows
    times = ev["decision_time"].values.astype("datetime64[ns]").astype(np.int64)
    a_vals = ev["A_surprise"].to_numpy(dtype=float)
    curr = ev["currency"].astype(str).to_numpy() if "currency" in ev.columns else np.array([""] * len(ev))
    types = ev["event_type"].astype(str).to_numpy()
    rows, prow = [], []
    last_by_type = {}
    for i in range(len(ev)):
        t = times[i]
        ts = pd.Timestamp(t, tz="UTC")
        lo = np.searchsorted(times, t - 24 * HOUR, side="left")
        before = np.arange(lo, i)
        before = before[times[before] < t]
        row = {"episode_id": hashlib.sha1(f"{types[i]}|{ts.isoformat()}".encode()).hexdigest()[:16],
               "event_type": types[i], "currency": curr[i], "published_at": ev["published_at"].iloc[i],
               "received_at": ev["received_at"].iloc[i], "decision_time": ts,
               "expectation_kind": ev["expectation_kind"].iloc[i], "raw_surprise": ev["raw_surprise"].iloc[i],
               "A_surprise": a_vals[i],
               "W_prior_events_24h": float(len(before)),
               "W_neighbour_surprise_24h": float(np.nansum(a_vals[before][curr[before] == curr[i]])) if len(before) else 0.0,
               "W_lag_A_same_type": last_by_type.get(types[i], np.nan)}
        row.update(_calendar(ts))
        anchor = t  # bars closed at or before t form the history
        entry = t   # first bar closing at or after t
        blk = _price_block(g, anchor, entry, short_h, long_h, placebo_h, history_h, barrier, ex)
        if np.isfinite(a_vals[i]):
            last_by_type[types[i]] = a_vals[i]
        if blk is None:
            continue
        row.update(blk)
        rows.append(row)
        # pseudo-event placebo: same type's A carried to t - offset where no release of the type happened
        tp = t - pseudo_offset_h * HOUR
        same = times[types == types[i]]
        if np.any(np.abs(same - tp) <= pseudo_exclusion_h * HOUR):
            ex["PSEUDO_EVENT_COLLIDES_WITH_RELEASE"] += 1
            continue
        pb = _price_block(g, tp, tp, short_h, long_h, placebo_h, history_h, barrier, Counter())
        if pb is None:
            continue
        pr = {k: row[k] for k in ("episode_id", "event_type", "currency", "A_surprise", "expectation_kind")}
        # event context recomputed at the pseudo instant: only releases decided strictly before tp
        plo = np.searchsorted(times, tp - 24 * HOUR, side="left")
        pbefore = np.arange(plo, np.searchsorted(times, tp, side="left"))
        pr["W_prior_events_24h"] = float(len(pbefore))
        pr["W_neighbour_surprise_24h"] = (float(np.nansum(a_vals[pbefore][curr[pbefore] == curr[i]]))
                                          if len(pbefore) else 0.0)
        same_before = np.where((types == types[i]) & (times < tp))[0]
        pr["W_lag_A_same_type"] = a_vals[same_before[-1]] if len(same_before) else np.nan
        pr.update(_calendar(pd.Timestamp(tp, tz="UTC")))
        pr.update(pb)
        pr["decision_time"] = pd.Timestamp(tp, tz="UTC")
        prow.append(pr)
    df = pd.DataFrame(rows)
    pdf = pd.DataFrame(prow)
    cuts = None
    if not df.empty and "W_rv_120h" in df:
        df["regime"], cuts = _regimes(df, "W_rv_120h")
        df["W_regime_code"] = df["regime"].map({"low_vol": 0.0, "mid_vol": 1.0, "high_vol": 2.0})
        if not pdf.empty:
            pdf["regime"], _ = _regimes(pdf, "W_rv_120h", cuts)
            pdf["W_regime_code"] = pdf["regime"].map({"low_vol": 0.0, "mid_vol": 1.0, "high_vol": 2.0})
    manifest = {"asset": asset, "train_end": str(g.train_end), "bars_digest": g.digest, "bars_rows_train": int(len(g.t)),
                "bar_minutes": bar_minutes, "bar_label": bar_label, "events_in": int(len(events)),
                "episodes": int(len(df)), "pseudo_episodes": int(len(pdf)),
                "expectation_rule": expectation, "scale_rule": "per event type, MAD*1.4826 (fallback sd) over TRAIN rows",
                "decision_time_rule": "max(published_at, received_at); entry = first bar closing at or after t",
                "history_rule": "bars closing at or before t", "outcome_rule": "elapsed hours from entry close",
                "barrier": barrier, "regime_cuts_W_rv_120h": cuts}
    return EpisodeSet(df, pdf, ex, manifest, {"scales": scales, "regime_cuts": cuts})


def asof_feature(feature, *, grid_times, value_col="value", time_col="timestamp", available_col=None,
                 availability_lag_h=0.0):
    """Value of a feature known at each grid instant (as-of join on availability, never on event time)."""
    f = feature.copy()
    f["_avail"] = _utc(f[available_col]) if available_col else _utc(f[time_col]) + pd.Timedelta(hours=availability_lag_h)
    f = f.dropna(subset=["_avail", value_col]).sort_values("_avail")
    av = f["_avail"].values.astype("datetime64[ns]").astype(np.int64)
    vals = f[value_col].to_numpy(dtype=float)
    idx = np.searchsorted(av, grid_times, side="right") - 1
    out = np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan)
    return out


def build_crossing_episodes(bars, *, train_end, feature_values=None, feature_name="feature", feature=None,
                            value_col="value", available_col=None, availability_lag_h=0.0, threshold_q=0.8,
                            band_q=0.6, direction="up", min_gap_h=24, control_stride_h=6, bar_minutes=60,
                            bar_label="open", short_h=SHORT_H, long_h=LONG_H, placebo_h=PLACEBO_H,
                            history_h=(24, 120), barrier=None, time_col="timestamp"):
    """Episodes where a feature crosses its TRAIN-fitted threshold (A=1) against comparable non-crossings (A=0).

    The decision instant is the bar close at which the crossing is first *available*.
    Controls: instants where the feature stayed on the pre-crossing side, with the previous
    value inside the same pre-band [q_band, q_threshold) so the pre-history is comparable.
    ``feature_values`` may be a callable(grid) -> values (e.g. a price-derived regime series).
    """
    barrier = {"tp_atr": 2.0, "sl_atr": 2.0, "timeout_h": 120, "atr_h": 24} if barrier is None else barrier
    g = PriceGrid(bars, train_end=train_end, bar_minutes=bar_minutes, bar_label=bar_label, time_col=time_col)
    grid = g.t
    if callable(feature_values):
        x = np.asarray(feature_values(g), dtype=float)
    elif feature is not None:
        x = asof_feature(feature, grid_times=grid, value_col=value_col, time_col=time_col,
                         available_col=available_col, availability_lag_h=availability_lag_h)
    else:
        raise EpisodeError("need feature or feature_values")
    ex = Counter()
    ok = np.isfinite(x)
    if ok.sum() < 50:
        raise EpisodeError(f"feature {feature_name} has too few TRAIN values ({int(ok.sum())})")
    sgn = 1.0 if direction == "up" else -1.0
    z = sgn * x
    thr = float(np.nanquantile(z, threshold_q))
    band = float(np.nanquantile(z, band_q))
    prev, now = z[:-1], z[1:]
    with np.errstate(invalid="ignore"):
        cross = (prev < thr) & (now >= thr)
        ctrl = (prev < thr) & (now < thr) & (prev >= band)
    cand_t = np.where(cross)[0] + 1
    treated, last = [], -np.inf
    for i in cand_t:
        if grid[i] - last >= min_gap_h * HOUR:
            treated.append(i)
            last = grid[i]
    tt = grid[treated] if treated else np.array([], dtype=np.int64)
    controls, lastc = [], -np.inf
    for i in np.where(ctrl)[0] + 1:
        if grid[i] - lastc < control_stride_h * HOUR:
            continue
        past = tt[tt <= grid[i]]  # only past crossings may exclude a control (no selection on the future)
        if len(past) and grid[i] - past[-1] < min_gap_h * HOUR:
            continue
        controls.append(i)
        lastc = grid[i]
    rows = []
    for arm, idxs in ((1.0, treated), (0.0, controls)):
        for i in idxs:
            t = grid[i]
            ts = pd.Timestamp(t, tz="UTC")
            blk = _price_block(g, t, t, short_h, long_h, placebo_h, history_h, barrier, ex)
            if blk is None:
                continue
            lag24 = x[i - 25] if i >= 25 else np.nan
            row = {"episode_id": hashlib.sha1(f"{feature_name}|{ts.isoformat()}".encode()).hexdigest()[:16],
                   "event_type": f"crossing:{feature_name}:{direction}", "decision_time": ts, "A_crossing": arm,
                   "W_x_prev": x[i - 1], "W_x_trend_24h": x[i - 1] - lag24 if np.isfinite(lag24) else np.nan}
            row.update(_calendar(ts))
            row.update(blk)
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True) if rows else pd.DataFrame()
    cuts = None
    if not df.empty:
        df["regime"], cuts = _regimes(df, "W_rv_120h")
        df["W_regime_code"] = df["regime"].map({"low_vol": 0.0, "mid_vol": 1.0, "high_vol": 2.0})
    manifest = {"feature": feature_name, "direction": direction, "train_end": str(g.train_end),
                "threshold_q": threshold_q, "threshold": thr * sgn, "band_q": band_q, "band": band * sgn,
                "treated": int((df.get("A_crossing", pd.Series(dtype=float)) == 1).sum()) if not df.empty else 0,
                "controls": int((df.get("A_crossing", pd.Series(dtype=float)) == 0).sum()) if not df.empty else 0,
                "bars_digest": g.digest, "availability_lag_h": availability_lag_h,
                "rule": "A=1 first available crossing of the TRAIN threshold (min gap); A=0 stayed below with prev in pre-band"}
    return EpisodeSet(df, pd.DataFrame(), ex, manifest, {"threshold": thr * sgn, "regime_cuts": cuts})


def realized_vol_series(hours=120):
    """Callable for ``build_crossing_episodes``: trailing realised vol of the asset itself (regime transitions)."""

    def fn(g):
        r = np.r_[np.nan, np.diff(g.logc)]
        s = pd.Series(r ** 2).rolling(hours, min_periods=hours // 2).sum()
        return np.sqrt(s.to_numpy())

    return fn
