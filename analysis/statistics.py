"""
Emergence Statistical Testing Framework

Rigorous emergence criteria, replacing simple slope judgments.
Includes: Mann-Kendall trend test, surrogate data method, cross-run distribution comparison.

References:
- Mann, H. B. (1945). Nonparametric tests against trend.
- Kendall, M. G. (1975). Rank correlation methods.
- Theiler, J. et al. (1992). Testing for nonlinearity in time series.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


# ============================================================
# Mann-Kendall Trend Test
# ============================================================

def mann_kendall_test(x: np.ndarray) -> Dict[str, float]:
    """
    Mann-Kendall trend test

    Null hypothesis H0: no monotonic trend in the series
    Alternative hypothesis H1: monotonic trend exists

    Returns:
        dict with 'tau' (Kendall's tau), 'p_value', 'trend' ('increasing'/'decreasing'/'none')
    """
    n = len(x)
    if n < 4:
        return {'tau': 0.0, 'p_value': 1.0, 'trend': 'none', 'significant': False}

    # Compute S statistic
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

    # Variance
    unique, counts = np.unique(x, return_counts=True)
    ties = counts[counts > 1]
    n_ties = len(ties)

    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    for t in ties:
        var_s -= t * (t - 1) * (2 * t + 5) / 18.0

    if var_s == 0:
        return {'tau': 0.0, 'p_value': 1.0, 'trend': 'none', 'significant': False}

    # Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # Kendall's tau
    tau = 2 * s / (n * (n - 1))

    # p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    if p_value < 0.05:
        trend = 'increasing' if tau > 0 else 'decreasing'
        significant = True
    else:
        trend = 'none'
        significant = False

    return {'tau': tau, 'p_value': p_value, 'z': z, 'trend': trend, 'significant': significant}


# ============================================================
# Surrogate Data Method -- Test for periodicity
# ============================================================

def generate_surrogate_data(x: np.ndarray, num_surrogates: int = 100) -> np.ndarray:
    """
    Generate surrogate datasets (IAAFT method - Iterative Amplitude Adjusted Fourier Transform)

    Preserves original data's amplitude spectrum, but randomizes phase.
    If original data's periodicity significantly exceeds surrogate data, periodicity is real.

    Simplified implementation: uses phase randomization
    """
    n = len(x)
    fft_x = np.fft.fft(x)
    magnitudes = np.abs(fft_x)
    phases = np.angle(fft_x)

    surrogates = np.zeros((num_surrogates, n))

    for i in range(num_surrogates):
        # Randomize phase (maintain conjugate symmetry)
        n_half = n // 2
        random_phases = np.random.uniform(0, 2 * np.pi, n_half)
        new_phases = np.zeros(n, dtype=complex)
        new_phases[0] = phases[0]  # 保持直流分量

        if n % 2 == 0:
            new_phases[1:n_half] = random_phases[:n_half - 1]
            new_phases[n_half] = phases[n_half]  # 奈奎斯特分量
            new_phases[n_half + 1:] = np.conj(new_phases[1:n_half][::-1])
        else:
            new_phases[1:n_half + 1] = random_phases
            new_phases[n_half + 1:] = np.conj(new_phases[1:n_half + 1][::-1])

        surrogate_fft = magnitudes * np.exp(1j * new_phases)
        surrogates[i] = np.real(np.fft.ifft(surrogate_fft))

    return surrogates


def test_periodicity(
    x: np.ndarray,
    num_surrogates: int = 100,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    使用替代数据法检验序列的周期性是否显著

    检验统计量：自相关的最大值（排除零滞后）

    Returns:
        dict with 'is_periodic', 'p_value', 'max_autocorr', 'surrogate_max_autocorr_dist'
    """
    n = len(x)
    if n < 20:
        return {'is_periodic': False, 'p_value': 1.0, 'reason': 'insufficient data'}

    # 原始数据的自相关最大值
    x_centered = x - x.mean()
    autocorr = np.correlate(x_centered, x_centered, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] > 0:
        autocorr_norm = autocorr / autocorr[0]
    else:
        return {'is_periodic': False, 'p_value': 1.0, 'reason': 'zero variance'}

    # 排除零滞后和前几个滞后
    min_lag = max(3, n // 20)
    max_lag = min(n // 2, 100)
    if min_lag >= max_lag:
        return {'is_periodic': False, 'p_value': 1.0, 'reason': 'too short for lag range'}

    original_max_ac = np.max(np.abs(autocorr_norm[min_lag:max_lag]))

    # 替代数据的自相关最大值
    surrogates = generate_surrogate_data(x, num_surrogates)
    surrogate_max_acs = []

    for surr in surrogates:
        surr_centered = surr - surr.mean()
        surr_ac = np.correlate(surr_centered, surr_centered, mode='full')
        surr_ac = surr_ac[len(surr_ac) // 2:]
        if surr_ac[0] > 0:
            surr_ac_norm = surr_ac / surr_ac[0]
            surrogate_max_acs.append(np.max(np.abs(surr_ac_norm[min_lag:max_lag])))
        else:
            surrogate_max_acs.append(0.0)

    surrogate_max_acs = np.array(surrogate_max_acs)

    # p 值：替代数据中超过原始数据的比例
    p_value = np.mean(surrogate_max_acs >= original_max_ac)

    # 检测到的周期（在原始自相关中的位置）
    peak_lag = min_lag + np.argmax(np.abs(autocorr_norm[min_lag:max_lag]))

    return {
        'is_periodic': p_value < significance_level,
        'p_value': p_value,
        'max_autocorr': original_max_ac,
        'surrogate_mean_max_ac': np.mean(surrogate_max_acs),
        'surrogate_std_max_ac': np.std(surrogate_max_acs),
        'detected_period': peak_lag,
        'significance_level': significance_level,
    }


    # ============================================================
# Cross-run distribution comparison
# ============================================================

def cross_run_comparison(
    results_list: List[Dict[str, Any]],
    metric: str,
    test: str = 'mann_whitney',
) -> Dict[str, Any]:
    """
    Compare a metric's distribution across multiple runs

    Args:
        results_list: list of results from multiple runs
        metric: metric name (e.g. 'final_gini', 'gini_slope', etc.)
        test: test method
    """
    values = []
    for r in results_list:
        ts = r.get('time_series', {})
        final = r.get('final_state', {})

        if metric == 'final_gini':
            values.append(final.get('gini', 0))
        elif metric == 'final_hhi':
            values.append(final.get('hhi', 0))
        elif metric == 'final_population':
            values.append(final.get('population', 0))
        elif metric == 'population_half_life':
            # 计算人口减半所需的时间步
            pop = ts.get('population', [])
            if len(pop) > 1:
                initial = pop[0]
                for i, p in enumerate(pop):
                    if p <= initial / 2:
                        values.append(i)
                        break
                else:
                    values.append(len(pop))  # 未减半
            else:
                values.append(0)
        else:
            # 默认取终值
            values.append(final.get(metric, 0))

    values = np.array(values)
    n = len(values)

    result = {
        'metric': metric,
        'n_runs': n,
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'min': np.min(values),
        'max': np.max(values),
        'values': values.tolist(),
    }

    # Test whether mean is significantly greater/less than zero
    if n >= 3:
        t_stat, p_value = stats.ttest_1samp(values, 0)
        result['t_stat'] = t_stat
        result['p_value_vs_zero'] = p_value
        result['significantly_positive'] = p_value < 0.05 and np.mean(values) > 0
        result['significantly_negative'] = p_value < 0.05 and np.mean(values) < 0

    return result


# ============================================================
# Crisis Event Detection
# ============================================================

def detect_crisis_events(
    production: np.ndarray,
    threshold_method: str = 'adaptive',
    window: int = 20,
) -> List[Dict[str, Any]]:
    """
    Detect economic crisis events

    Define crisis: output significantly below recent mean

    Args:
        production: production time series
        threshold_method: 'adaptive' (adaptive threshold) or 'fixed' (fixed threshold)
        window: sliding window for computing baseline

    Returns:
        List of crisis events
    """
    n = len(production)
    if n < window * 2:
        return []

    crises = []
    in_crisis = False
    crisis_start = None

    for i in range(window, n):
        # Baseline = mean of previous window
        baseline = np.mean(production[max(0, i - window):i])
        baseline_std = np.std(production[max(0, i - window):i])

        if baseline_std > 0:
            # Output below baseline by 1.5 std -> crisis
            deviation = (baseline - production[i]) / baseline_std
        else:
            deviation = 0

        if deviation > 1.5 and not in_crisis:
            in_crisis = True
            crisis_start = i
        elif deviation <= 0.5 and in_crisis:
            in_crisis = False
            if crisis_start is not None:
                crises.append({
                    'start_tick': crisis_start,
                    'end_tick': i,
                    'duration': i - crisis_start,
                    'max_deviation': deviation,
                    'baseline_production': baseline,
                    'crisis_trough': np.min(production[crisis_start:i]),
                })
                crisis_start = None

    # Handle unended crises
    if in_crisis and crisis_start is not None:
        crises.append({
            'start_tick': crisis_start,
            'end_tick': n,
            'duration': n - crisis_start,
            'max_deviation': 0,
            'baseline_production': np.mean(production[max(0, n - window):n]),
            'crisis_trough': np.min(production[crisis_start:]),
        })

    return crises
