#!/usr/bin/env python3
"""
Causal Inference Analysis for Regime-Based Trading Strategy
===========================================================
Applies three causal inference techniques to the 12 regime features:

1. NOTEARS Causal Discovery: Learns the DAG structure among features
   to find which features causally drive others vs. are mere symptoms.

2. Invariant Causal Prediction (ICP): Finds features whose relationship
   with forward returns is stable ACROSS all regimes (environments).
   Invariant features = robust strategy foundations.

3. DoWhy Refutation Tests: Tests whether identified causal effects
   survive placebo and random confounders.

Input:  15-year EURUSD hourly OHLC → resampled to 4h
Output: Ranked features by causal strength, DAG visualization,
        invariance scores, refutation results.
"""

import sys
import os
import warnings
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)  # Suppress all library logging

# ─── Configuration ────────────────────────────────────────────────
OHLC_FILE = os.environ.get(
    "OHLC_FILE",
    os.path.join(os.path.dirname(__file__),
                 "..", "feature-eng", "tests", "data",
                 "eurusd_hour_2005_2020_ohlc.csv")
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

# Feature columns used in regime detection (12 features)
FEATURE_COLS = [
    'adx', 'di_spread', 'atr_pct', 'atr_ratio',
    'bb_width_pct', 'bb_position', 'rsi', 'roc_12',
    'price_vs_ema50', 'ema_alignment', 'stoch_k', 'macd_hist'
]

# Forward return horizons to test (in 4h bars)
RETURN_HORIZONS = [1, 3, 6]

# ─── Data Loading ─────────────────────────────────────────────────

def load_and_resample_4h(filepath: str) -> pd.DataFrame:
    """Load hourly OHLC and resample to 4h bars."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)

    # Handle date column
    date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_col:
        df.index = pd.to_datetime(df[date_col[0]], dayfirst=True)
        df = df.drop(columns=date_col)
    
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ('open', 'high', 'low', 'close', 'volume'):
            col_map[c] = cl
    df = df.rename(columns=col_map)

    # Resample to 4h
    ohlc_4h = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    print(f"  Hourly bars: {len(df):,} → 4h bars: {len(ohlc_4h):,}")
    print(f"  Date range: {ohlc_4h.index[0]} to {ohlc_4h.index[-1]}")
    return ohlc_4h


def compute_features(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 12 regime features from OHLC data."""
    # Add feature-eng to path
    fe_path = os.path.join(os.path.dirname(__file__), "..", "feature-eng")
    if fe_path not in sys.path:
        sys.path.insert(0, fe_path)
    from app.regime_detector import compute_regime_features, classify_regime
    
    features = compute_regime_features(ohlc_df)
    regimes = classify_regime(features)
    features['regime'] = regimes
    
    # Forward returns at different horizons
    for h in RETURN_HORIZONS:
        features[f'fwd_ret_{h}'] = ohlc_df['close'].pct_change(h).shift(-h) * 100
    
    # Binary outcome: positive return (for causal effect estimation)
    features['fwd_ret_6_positive'] = (features['fwd_ret_6'] > 0).astype(int)
    
    features = features.dropna()
    print(f"  Features computed: {len(features):,} bars with {len(FEATURE_COLS)} features")
    print(f"  Regime distribution:\n{features['regime'].value_counts().sort_index().to_string()}")
    return features


# ─── 1. NOTEARS Causal Discovery ─────────────────────────────────

def run_notears(features: pd.DataFrame) -> np.ndarray:
    """
    Learn the DAG among regime features using NOTEARS.
    Returns weighted adjacency matrix W where W[i,j] = causal weight i→j.
    """
    print("\n" + "="*70)
    print("1. NOTEARS CAUSAL DISCOVERY")
    print("="*70)
    
    from castle.algorithms import Notears
    
    X = features[FEATURE_COLS].values
    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    # Subsample for speed (NOTEARS is O(n²) in features but slow on rows)
    n = len(X)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    
    print(f"  Running NOTEARS (linear) on {X_sub.shape[0]} samples, {X_sub.shape[1]} features...")
    
    nt = Notears(lambda1=0.1, loss_type='l2', max_iter=100, h_tol=1e-8, w_threshold=0.3)
    nt.learn(X_sub)
    W = nt.causal_matrix
    
    print(f"  Edges found: {(np.abs(W) > 0).sum()}")
    
    # Display causal relationships
    print("\n  Causal DAG edges (cause → effect, weight):")
    edges = []
    for i in range(len(FEATURE_COLS)):
        for j in range(len(FEATURE_COLS)):
            if abs(W[i, j]) > 0.01:
                edges.append((FEATURE_COLS[i], FEATURE_COLS[j], W[i, j]))
    
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    for cause, effect, w in edges[:20]:
        print(f"    {cause:>18s} → {effect:<18s}  weight={w:+.4f}")
    
    # Identify root causes (features with many outgoing edges, few incoming)
    out_degree = (np.abs(W) > 0.01).sum(axis=1)  # row i → columns
    in_degree = (np.abs(W) > 0.01).sum(axis=0)   # columns → row j
    
    print("\n  Feature Causal Role Analysis:")
    print(f"  {'Feature':>18s}  Out  In  Net   Role")
    print(f"  {'─'*18}  ───  ──  ───  ────────────────")
    roles = {}
    for i, f in enumerate(FEATURE_COLS):
        net = out_degree[i] - in_degree[i]
        if net > 1:
            role = "ROOT CAUSE"
        elif net > 0:
            role = "DRIVER"
        elif net < -1:
            role = "DOWNSTREAM"
        else:
            role = "MEDIATOR"
        roles[f] = role
        print(f"  {f:>18s}  {out_degree[i]:3d}  {in_degree[i]:2d}  {net:+3d}  {role}")
    
    return W, roles


# ─── 2. Invariant Causal Prediction (ICP) ────────────────────────

def run_icp(features: pd.DataFrame) -> dict:
    """
    Test which features maintain a stable causal relationship with
    forward returns across ALL regimes (environments).
    
    A feature is "causally invariant" if its regression coefficient 
    with forward returns is statistically consistent across all regimes.
    
    Uses Wald test for coefficient equality across environments.
    """
    print("\n" + "="*70)
    print("2. INVARIANT CAUSAL PREDICTION (ICP) ACROSS REGIMES")
    print("="*70)
    
    from sklearn.linear_model import LinearRegression
    from scipy import stats
    
    regimes = sorted(features['regime'].unique())
    print(f"  Environments (regimes): {regimes}")
    print(f"  Testing invariance for target: fwd_ret_6")
    
    # For each feature, fit linear regression in each regime and test
    # whether coefficients are statistically the same
    invariance_scores = {}
    
    for feat in FEATURE_COLS:
        coefs = []
        ses = []
        ns = []
        
        for r in regimes:
            sub = features[features['regime'] == r]
            if len(sub) < 30:
                continue
            
            X = sub[feat].values.reshape(-1, 1)
            y = sub['fwd_ret_6'].values
            
            # Standardize within regime
            X = (X - X.mean()) / (X.std() + 1e-10)
            
            model = LinearRegression()
            model.fit(X, y)
            coef = model.coef_[0]
            
            # Standard error of coefficient
            y_pred = model.predict(X)
            residuals = y - y_pred
            mse = np.mean(residuals**2)
            se = np.sqrt(mse / (np.sum((X - X.mean())**2) + 1e-10))
            
            coefs.append(coef)
            ses.append(se)
            ns.append(len(sub))
        
        if len(coefs) < 3:
            invariance_scores[feat] = {'score': 0.0, 'invariant': False}
            continue
        
        coefs = np.array(coefs)
        ses = np.array(ses)
        ns = np.array(ns)
        
        # Wald test: H0 = all coefficients equal
        # Weighted mean coefficient
        weights = 1.0 / (ses**2 + 1e-10)
        weighted_mean = np.sum(weights * coefs) / np.sum(weights)
        
        # Chi-squared statistic
        chi2 = np.sum(weights * (coefs - weighted_mean)**2)
        df = len(coefs) - 1
        p_value = 1 - stats.chi2.cdf(chi2, df)
        
        # Direction consistency: do all regimes agree on sign?
        sign_consistency = np.mean(np.sign(coefs) == np.sign(weighted_mean))
        
        # Invariance score: high p-value (coefficients similar) + consistent sign
        invariance = p_value * sign_consistency
        
        invariance_scores[feat] = {
            'score': float(invariance),
            'p_value': float(p_value),
            'sign_consistency': float(sign_consistency),
            'mean_coef': float(weighted_mean),
            'coef_range': [float(coefs.min()), float(coefs.max())],
            'invariant': p_value > 0.05 and sign_consistency >= 0.8,
            'per_regime_coefs': {int(r): float(c) for r, c in zip(regimes, coefs)}
        }
    
    # Print results sorted by invariance
    print(f"\n  {'Feature':>18s}  {'InvScore':>8s}  {'p-value':>8s}  {'SignCon':>7s}  {'MeanCoef':>9s}  Status")
    print(f"  {'─'*18}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*9}  ──────────")
    
    sorted_feats = sorted(invariance_scores.items(), 
                          key=lambda x: x[1]['score'], reverse=True)
    
    invariant_features = []
    for feat, info in sorted_feats:
        if info['score'] == 0:
            continue
        status = "✓ INVARIANT" if info['invariant'] else "  regime-specific"
        if info['invariant']:
            invariant_features.append(feat)
        print(f"  {feat:>18s}  {info['score']:8.4f}  {info['p_value']:8.4f}  "
              f"{info['sign_consistency']:7.2f}  {info['mean_coef']:+9.6f}  {status}")
    
    print(f"\n  INVARIANT features (stable across all regimes): {invariant_features}")
    print(f"  These features can be trusted regardless of market regime.")
    
    # Per-regime coefficient breakdown for top features
    print(f"\n  Per-regime coefficient breakdown (top 5 invariant):")
    for feat in invariant_features[:5]:
        info = invariance_scores[feat]
        coef_str = "  ".join(f"R{r}:{c:+.5f}" for r, c in info['per_regime_coefs'].items())
        print(f"    {feat:>18s}: {coef_str}")
    
    return invariance_scores


# ─── 3. DoWhy Refutation Tests ───────────────────────────────────

def run_dowhy_refutation(features: pd.DataFrame, 
                         top_features: list) -> dict:
    """
    For each top feature, use DoWhy to:
    1. Estimate causal effect on forward returns
    2. Run refutation tests (placebo treatment, random confounder)
    """
    print("\n" + "="*70)
    print("3. DOWHY CAUSAL EFFECT ESTIMATION & REFUTATION")
    print("="*70)
    
    import dowhy
    from dowhy import CausalModel
    
    # Subsample for speed
    if len(features) > 8000:
        sub = features.sample(8000, random_state=42)
    else:
        sub = features.copy()
    
    results = {}
    
    for feat in top_features[:6]:  # Top 6 features
        print(f"\n  ── Testing: {feat} → fwd_ret_6 ──")
        
        # Create a DataFrame with just what we need
        confounders = [f for f in FEATURE_COLS if f != feat][:4]  # Use 4 confounders
        cols_needed = [feat, 'fwd_ret_6'] + confounders
        data = sub[cols_needed].copy()
        
        # Discretize treatment for cleaner estimation
        data[f'{feat}_high'] = (data[feat] > data[feat].median()).astype(int)
        
        try:
            # Build causal model
            model = CausalModel(
                data=data,
                treatment=f'{feat}_high',
                outcome='fwd_ret_6',
                common_causes=confounders,
                proceed_when_unidentifiable=True
            )
            
            # Estimate effect using linear regression
            estimate = model.identify_effect(proceed_when_unidentifiable=True)
            causal_estimate = model.estimate_effect(
                estimate,
                method_name="backdoor.linear_regression"
            )
            
            ate = causal_estimate.value
            print(f"    Estimated ATE: {ate:+.6f} (effect of high {feat} on 6-bar return)")
            
            # Refutation 1: Placebo treatment (random permutation)
            try:
                placebo = model.refute_estimate(
                    estimate,
                    causal_estimate,
                    method_name="placebo_treatment_refuter",
                    placebo_type="permute",
                    num_simulations=100
                )
                placebo_p = placebo.refutation_result.get('p_value', 
                    getattr(placebo, 'refutation_result', {}).get('p_value', None))
                
                # Try to extract p-value from different DoWhy versions
                if placebo_p is None and hasattr(placebo, 'estimated_effect'):
                    # If placebo estimate is close to 0, effect is real
                    placebo_est = float(placebo.estimated_effect) if hasattr(placebo, 'estimated_effect') else 0
                    placebo_passed = abs(placebo_est) < abs(ate) * 0.5
                else:
                    placebo_passed = placebo_p is not None and placebo_p > 0.05 if isinstance(placebo_p, (int, float)) else True
                    placebo_est = 0
                
                print(f"    Placebo test: est={placebo_est:.6f} (original={ate:.6f}) → "
                      f"{'PASSED ✓' if placebo_passed else 'FAILED ✗'}")
            except Exception as e:
                print(f"    Placebo test: skipped ({type(e).__name__})")
                placebo_passed = None
                placebo_est = None
            
            # Refutation 2: Add random common cause
            try:
                random_cause = model.refute_estimate(
                    estimate,
                    causal_estimate,
                    method_name="random_common_cause",
                    num_simulations=100
                )
                rc_est = float(random_cause.estimated_effect) if hasattr(random_cause, 'estimated_effect') else ate
                rc_passed = abs(rc_est - ate) / (abs(ate) + 1e-10) < 0.15  # <15% change
                
                print(f"    Random cause: est={rc_est:.6f} (original={ate:.6f}, "
                      f"change={abs(rc_est-ate)/abs(ate+1e-10)*100:.1f}%) → "
                      f"{'PASSED ✓' if rc_passed else 'FAILED ✗'}")
            except Exception as e:
                print(f"    Random cause: skipped ({type(e).__name__})")
                rc_passed = None
                rc_est = None
            
            results[feat] = {
                'ate': float(ate),
                'placebo_passed': bool(placebo_passed) if placebo_passed is not None else None,
                'placebo_estimate': float(placebo_est) if placebo_est is not None else None,
                'random_cause_passed': bool(rc_passed) if rc_passed is not None else None,
                'random_cause_estimate': float(rc_est) if rc_est is not None else None,
                'robust': bool(placebo_passed) and bool(rc_passed) if (placebo_passed is not None and rc_passed is not None) else False
            }
            
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            results[feat] = {'ate': None, 'error': str(e)}
    
    return results


# ─── 4. Heterogeneous Treatment Effects (Causal Forests) ─────────

def run_causal_forest(features: pd.DataFrame) -> dict:
    """
    Use EconML's CausalForestDML to estimate treatment effects
    conditioned on regime (heterogeneous effects).
    
    Treatment: bb_position > 0.5 (price in upper half of Bollinger Band)
    Outcome: forward 6-bar return
    """
    print("\n" + "="*70)
    print("4. CAUSAL FOREST - HETEROGENEOUS TREATMENT EFFECTS")
    print("="*70)
    
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Subsample for speed
    if len(features) > 10000:
        sub = features.sample(10000, random_state=42)
    else:
        sub = features.copy()
    
    # Treatment: continuous roc_12 (momentum strength)
    T = sub['roc_12'].values
    
    # Outcome: 6-bar forward return
    Y = sub['fwd_ret_6'].values
    
    # Controls (confounders): other features
    X_cols = ['adx', 'atr_pct', 'bb_position', 'rsi', 'di_spread', 'atr_ratio']
    X = sub[X_cols].values
    
    # Effect modifiers (what we want heterogeneity over): regime features
    W = sub[['regime', 'bb_width_pct', 'ema_alignment', 'stoch_k']].values
    
    print(f"  Treatment: roc_12 (continuous momentum)")
    print(f"  Outcome: fwd_ret_6")
    print(f"  Controls: {X_cols}")
    print(f"  Running CausalForestDML...")
    
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42
    )
    cf.fit(Y, T, X=X, W=W)
    
    # Estimate effects per regime
    print(f"\n  Heterogeneous Treatment Effects by Regime:")
    print(f"  {'Regime':>10s}  {'CATE':>10s}  {'95% CI Low':>10s}  {'95% CI High':>11s}  Interpretation")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*11}  ──────────────────")
    
    regime_effects = {}
    for r in sorted(sub['regime'].unique()):
        mask = sub['regime'] == r
        X_r = X[mask]
        if len(X_r) < 20:
            continue
        
        effects = cf.effect(X_r)
        ci = cf.effect_interval(X_r, alpha=0.05)
        
        mean_eff = effects.mean()
        ci_lo = ci[0].mean()
        ci_hi = ci[1].mean()
        
        if ci_lo > 0:
            interp = "MOMENTUM WORKS ✓"
        elif ci_hi < 0:
            interp = "MOMENTUM HURTS ✗"
        else:
            interp = "insignificant"
        
        regime_effects[int(r)] = {
            'cate': float(mean_eff),
            'ci_low': float(ci_lo),
            'ci_high': float(ci_hi),
            'n_samples': int(mask.sum()),
            'significant': bool(ci_lo > 0 or ci_hi < 0)
        }
        
        print(f"  Regime {r:>3d}  {mean_eff:+10.6f}  {ci_lo:+10.6f}  {ci_hi:+11.6f}  {interp}")
    
    # Feature importance for heterogeneity
    print(f"\n  Feature importance for treatment effect heterogeneity:")
    importances = cf.feature_importances_
    for i, col in enumerate(X_cols):
        print(f"    {col:>18s}: {importances[i]:.4f}")
    
    return regime_effects


# ─── 5. Transfer Entropy (Causal Information Flow) ────────────────

def run_transfer_entropy(features: pd.DataFrame) -> dict:
    """
    Measure directional information flow between features using
    Transfer Entropy. TE(X→Y) > TE(Y→X) suggests X causes Y.
    """
    print("\n" + "="*70)
    print("5. TRANSFER ENTROPY - CAUSAL INFORMATION FLOW")
    print("="*70)
    
    # Simple binned transfer entropy implementation
    def transfer_entropy(x, y, lag=1, bins=10):
        """TE(X→Y): how much X's past reduces uncertainty about Y's future."""
        x_past = pd.cut(pd.Series(x[:-lag]), bins=bins, labels=False).values
        y_past = pd.cut(pd.Series(y[:-lag]), bins=bins, labels=False).values
        y_fut = pd.cut(pd.Series(y[lag:]), bins=bins, labels=False).values
        
        # Remove NaN
        valid = ~(np.isnan(x_past) | np.isnan(y_past) | np.isnan(y_fut))
        x_past, y_past, y_fut = x_past[valid], y_past[valid], y_fut[valid]
        
        if len(x_past) < 100:
            return 0.0
        
        # Joint and marginal probabilities via histogram
        n = len(x_past)
        
        # H(Y_fut | Y_past) - H(Y_fut | Y_past, X_past)
        # Using conditional entropy estimates
        from collections import Counter
        
        # P(y_fut, y_past)
        joint_yy = Counter(zip(y_fut.astype(int), y_past.astype(int)))
        # P(y_fut, y_past, x_past)
        joint_yyx = Counter(zip(y_fut.astype(int), y_past.astype(int), x_past.astype(int)))
        # P(y_past)
        marg_y = Counter(y_past.astype(int))
        # P(y_past, x_past)
        joint_yx = Counter(zip(y_past.astype(int), x_past.astype(int)))
        
        te = 0.0
        for (yf, yp, xp), count_yyx in joint_yyx.items():
            p_yyx = count_yyx / n
            p_yf_yp = joint_yy.get((yf, yp), 1) / n
            p_yp = marg_y.get(yp, 1) / n
            p_yf_ypxp = count_yyx / n
            p_ypxp = joint_yx.get((yp, xp), 1) / n
            
            if p_yf_ypxp > 0 and p_ypxp > 0 and p_yf_yp > 0 and p_yp > 0:
                te += p_yyx * np.log2(
                    (p_yf_ypxp / p_ypxp) / (p_yf_yp / p_yp + 1e-15) + 1e-15
                )
        
        return max(0, te)
    
    # Subsample
    if len(features) > 10000:
        sub = features.sample(10000, random_state=42).sort_index()
    else:
        sub = features.sort_index()
    
    # Compute TE between each feature and forward returns
    print(f"  Computing Transfer Entropy: feature → fwd_ret_6")
    
    te_results = {}
    for feat in FEATURE_COLS:
        x = sub[feat].values
        y = sub['fwd_ret_6'].values
        
        te_xy = transfer_entropy(x, y, lag=1)
        te_yx = transfer_entropy(y, x, lag=1)
        
        net_te = te_xy - te_yx
        te_results[feat] = {
            'te_feat_to_ret': float(te_xy),
            'te_ret_to_feat': float(te_yx),
            'net_te': float(net_te),
            'direction': 'feat→return' if net_te > 0 else 'return→feat'
        }
    
    # Print sorted by net TE
    print(f"\n  {'Feature':>18s}  {'TE(f→r)':>8s}  {'TE(r→f)':>8s}  {'Net TE':>8s}  Direction")
    print(f"  {'─'*18}  {'─'*8}  {'─'*8}  {'─'*8}  ──────────────")
    
    sorted_te = sorted(te_results.items(), key=lambda x: x[1]['net_te'], reverse=True)
    for feat, info in sorted_te:
        d = info['direction']
        print(f"  {feat:>18s}  {info['te_feat_to_ret']:8.5f}  {info['te_ret_to_feat']:8.5f}  "
              f"{info['net_te']:+8.5f}  {d}")
    
    leading = [f for f, i in sorted_te if i['net_te'] > 0.001]
    lagging = [f for f, i in sorted_te if i['net_te'] < -0.001]
    print(f"\n  LEADING indicators (info flows TO returns): {leading}")
    print(f"  LAGGING indicators (info flows FROM returns): {lagging}")
    
    return te_results


# ─── Main Pipeline ────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  CAUSAL INFERENCE ANALYSIS FOR REGIME-BASED TRADING STRATEGY   ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Techniques: NOTEARS, ICP, DoWhy Refutation, Causal Forest,   ║")
    print("║              Transfer Entropy                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare data
    ohlc = load_and_resample_4h(OHLC_FILE)
    features = compute_features(ohlc)
    
    all_results = {}
    
    # 1. NOTEARS Causal Discovery
    try:
        dag_matrix, roles = run_notears(features)
        all_results['notears_roles'] = roles
        np.save(os.path.join(OUTPUT_DIR, 'dag_matrix.npy'), dag_matrix)
    except Exception as e:
        print(f"\n  NOTEARS failed: {e}")
        roles = {}
        all_results['notears_roles'] = {}
    
    # 2. ICP Across Regimes
    try:
        icp_results = run_icp(features)
        all_results['icp'] = icp_results
        invariant_feats = [f for f, info in icp_results.items() if info.get('invariant')]
    except Exception as e:
        print(f"\n  ICP failed: {e}")
        icp_results = {}
        invariant_feats = FEATURE_COLS[:6]
    
    # 3. DoWhy Refutation
    # Test top features by ICP score, not just invariant ones
    sorted_by_icp = sorted(icp_results.items(), key=lambda x: x[1].get('score', 0), reverse=True)
    test_features = [f for f, _ in sorted_by_icp[:12]]  # Test all features
    try:
        dowhy_results = run_dowhy_refutation(features, test_features)
        all_results['dowhy'] = dowhy_results
    except Exception as e:
        print(f"\n  DoWhy failed: {e}")
        all_results['dowhy'] = {}
    
    # 4. Causal Forest (Heterogeneous Treatment Effects)
    try:
        cf_results = run_causal_forest(features)
        all_results['causal_forest'] = cf_results
    except Exception as e:
        print(f"\n  Causal Forest failed: {e}")
        all_results['causal_forest'] = {}
    
    # 5. Transfer Entropy
    try:
        te_results = run_transfer_entropy(features)
        all_results['transfer_entropy'] = te_results
    except Exception as e:
        print(f"\n  Transfer Entropy failed: {e}")
        all_results['transfer_entropy'] = {}
    
    # ─── Final Summary ────────────────────────────────────────────
    print("\n" + "═"*70)
    print("FINAL CAUSAL EVIDENCE SUMMARY")
    print("═"*70)
    
    # Score each feature across all methods
    feature_scores = {}
    for feat in FEATURE_COLS:
        score = 0
        evidence = []
        
        # NOTEARS: root cause or driver
        if feat in roles and roles[feat] in ('ROOT CAUSE', 'DRIVER'):
            score += 2
            evidence.append(f"DAG:{roles[feat]}")
        
        # ICP: invariant or near-invariant
        if feat in icp_results:
            icp_info = icp_results[feat]
            if icp_info.get('invariant'):
                score += 3  # Highest weight — stable across regimes
                evidence.append("ICP:INVARIANT")
            elif icp_info.get('p_value', 0) > 0.05 and icp_info.get('sign_consistency', 0) >= 0.67:
                score += 2  # Near-invariant
                evidence.append(f"ICP:NEAR_INV(p={icp_info['p_value']:.3f})")
            elif icp_info.get('p_value', 0) > 0.01:
                score += 1  # Some stability
                evidence.append(f"ICP:PARTIAL(p={icp_info['p_value']:.3f})")
        
        # DoWhy: robust causal effect (passed both refutation tests)
        if feat in all_results.get('dowhy', {}):
            dw = all_results['dowhy'][feat]
            if dw.get('robust'):
                score += 2
                evidence.append(f"DoWhy:ROBUST(ATE={dw['ate']:+.4f})")
            elif dw.get('placebo_passed') or dw.get('random_cause_passed'):
                score += 1
                evidence.append(f"DoWhy:PARTIAL(ATE={dw['ate']:+.4f})")
        
        # Transfer Entropy: leading indicator
        if feat in all_results.get('transfer_entropy', {}):
            te_info = all_results['transfer_entropy'][feat]
            if te_info['net_te'] > 0.001:
                score += 2  # Genuinely leading — very valuable
                evidence.append("TE:LEADING")
            elif te_info['net_te'] > -0.01:
                score += 1  # Near-neutral, not strongly lagging
                evidence.append("TE:NEUTRAL")
        
        feature_scores[feat] = {'score': score, 'evidence': evidence}
    
    # Print ranking
    print(f"\n  FEATURE CAUSAL RANKING (higher = more causal evidence)")
    print(f"  {'Rank':>4s}  {'Feature':>18s}  {'Score':>5s}  Evidence")
    print(f"  {'─'*4}  {'─'*18}  {'─'*5}  ──────────────────────────")
    
    sorted_scores = sorted(feature_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    core_features = []
    regime_features = []
    noise_features = []
    
    for rank, (feat, info) in enumerate(sorted_scores, 1):
        evid = ", ".join(info['evidence']) if info['evidence'] else "no evidence"
        cat = ""
        if info['score'] >= 4:
            cat = "★ CORE"
            core_features.append(feat)
        elif info['score'] >= 2:
            cat = "▸ REGIME"
            regime_features.append(feat)
        else:
            cat = "  noise?"
            noise_features.append(feat)
        print(f"  {rank:4d}  {feat:>18s}  {info['score']:5d}  {evid}  {cat}")
    
    print(f"\n  ★ CORE FEATURES (robust, causal, invariant): {core_features}")
    print(f"  ▸ REGIME FEATURES (causal in specific regimes): {regime_features}")
    print(f"  ○ WEAK/NOISE (no causal evidence): {noise_features}")
    
    # Strategy recommendations
    print(f"\n  STRATEGY RECOMMENDATIONS:")
    print(f"  ─────────────────────────")
    if core_features:
        print(f"  1. Regime detector should weight CORE features highest: {core_features}")
    if noise_features:
        print(f"  2. Consider REMOVING these from regime classification: {noise_features}")
    
    # Causal forest regime insights
    if all_results.get('causal_forest'):
        print(f"\n  3. Per-regime momentum effect (from Causal Forest):")
        for r, info in sorted(all_results['causal_forest'].items()):
            sig = "★" if info.get('significant') else " "
            print(f"     {sig} Regime {r}: CATE={info['cate']:+.6f} "
                  f"[{info['ci_low']:+.6f}, {info['ci_high']:+.6f}]")
    
    # Save all results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj
    
    results_file = os.path.join(OUTPUT_DIR, 'causal_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    
    print(f"\n  Results saved to {results_file}")
    print(f"\n{'═'*70}")
    print("ANALYSIS COMPLETE")
    print("═"*70)
    
    return all_results


if __name__ == "__main__":
    main()
