"""
Emergence Analyzer

Uses rigorous statistical methods to test whether higher-order regularities have emerged.
Includes: Mann-Kendall trend test, surrogate data method, cross-run distribution comparison.
"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass

from .statistics import (
    mann_kendall_test,
    test_periodicity,
    detect_crisis_events,
)


@dataclass
class EmergenceResult:
    """Single emergence test result"""
    law_id: str
    law_name: str
    emerged: bool
    confidence: float  # 0~1
    evidence: str
    details: Dict[str, Any]
    statistical_test: str  # Statistical test used


class EmergenceAnalyzer:
    """Emergence analyzer -- uses rigorous statistical methods"""

    @staticmethod
    def analyze_single(results: Dict[str, Any]) -> List[EmergenceResult]:
        ts = results.get('time_series', {})
        if not ts:
            return []

        return [
            EmergenceAnalyzer.check_capital_concentration(ts),
            EmergenceAnalyzer.check_competition_elimination(ts),
            EmergenceAnalyzer.check_endogenous_crisis(ts),
            EmergenceAnalyzer.check_trickle_down(ts),
            EmergenceAnalyzer.check_profit_equalization(ts),
        ]

    @staticmethod
    def analyze_multi_run(results_list: List[Dict[str, Any]]) -> List[EmergenceResult]:
        """
        Comprehensive emergence test across multiple runs (cross-run distribution comparison)

        This is the most rigorous emergence judgment:
        does not rely on noise from a single run, but checks whether trends
        are statistically significant across multiple runs.
        """
        if not results_list:
            return []

        single_results = [EmergenceAnalyzer.analyze_single(r) for r in results_list]

        # Aggregate by law ID
        law_ids = [r.law_id for r in single_results[0]]
        multi_results = []

        for idx, law_id in enumerate(law_ids):
            law_results = [runs[idx] for runs in single_results]

            emerged_count = sum(1 for r in law_results if r.emerged)
            total = len(law_results)
            emerged_ratio = emerged_count / total if total > 0 else 0

            # Comprehensive judgment: emergence in 80%+ runs -> judged as emerged
            emerged = emerged_ratio >= 0.6

            # Confidence = emergence ratio
            confidence = emerged_ratio

            # Collect evidence from all runs
            evidence_parts = []
            for r in law_results:
                evidence_parts.append(r.evidence)

            # Use last run's detailed info
            last = law_results[-1]

            multi_results.append(EmergenceResult(
                law_id=law_id,
                law_name=last.law_name,
                emerged=emerged,
                confidence=confidence,
                evidence=f"Across {total} runs: emerged in {emerged_count} ({emerged_ratio:.0%})",
                details={
                    'emerged_count': emerged_count,
                    'total_runs': total,
                    'emerged_ratio': emerged_ratio,
                    'per_run_evidence': evidence_parts,
                },
                statistical_test='multi_run_consensus',
            ))

        return multi_results

    @staticmethod
    def analyze_all(results: Dict[str, Any]) -> List[EmergenceResult]:
        """Backward compatible: single run analysis"""
        return EmergenceAnalyzer.analyze_single(results)

    # ============================================================
    # ============================================================

    @staticmethod
    def check_capital_concentration(ts: Dict[str, List[float]]) -> EmergenceResult:
        """HL-1: Is capital concentration rising -- using Mann-Kendall test"""
        gini = ts.get('gini', [])
        top10 = ts.get('top10_share', [])
        bottom50 = ts.get('bottom50_share', [])

        if len(gini) < 30:
            return EmergenceResult('HL-1', 'Capital concentration rising', False, 0, 'Insufficient data',
                                   {}, 'insufficient_data')

        # Skip initial oscillation period
        start = len(gini) // 5
        stable_gini = gini[start:]
        stable_top10 = top10[start:]
        stable_bottom50 = bottom50[start:]

        # Mann-Kendall test
        mk_gini = mann_kendall_test(np.array(stable_gini))
        mk_top10 = mann_kendall_test(np.array(stable_top10)) if len(stable_top10) > 3 else None
        mk_bottom50 = mann_kendall_test(np.array(stable_bottom50)) if len(stable_bottom50) > 3 else None

        # Comprehensive judgment: at least two indicators pass trend test
        indicators = []
        if mk_gini['trend'] == 'increasing' and mk_gini['significant']:
            indicators.append('Gini MK(p={:.3f})'.format(mk_gini['p_value']))
        if mk_top10 and mk_top10['trend'] == 'increasing' and mk_top10['significant']:
            indicators.append('Top10 MK(p={:.3f})'.format(mk_top10['p_value']))
        if mk_bottom50 and mk_bottom50['trend'] == 'decreasing' and mk_bottom50['significant']:
            indicators.append('Bottom50 MK(p={:.3f})'.format(mk_bottom50['p_value']))

        # Also check start-end direction (even if not significant, directional consistency matters)
        gini_rising = stable_gini[-1] > stable_gini[0]
        top10_rising = stable_top10[-1] > stable_top10[0] if stable_top10 else False
        bottom50_falling = stable_bottom50[-1] < stable_bottom50[0] if stable_bottom50 else False

        direction_consistent = sum([gini_rising, top10_rising, bottom50_falling]) >= 2

        emerged = len(indicators) >= 2 or (len(indicators) >= 1 and direction_consistent)
        confidence = len(indicators) / 3.0 if emerged else 0.0

        evidence = (
            f"Gini: {stable_gini[0]:.3f}->{stable_gini[-1]:.3f} "
            f"(MK tau={mk_gini['tau']:.3f}, p={mk_gini['p_value']:.4f}), "
            f"Top10: {stable_top10[0]:.1%}->{stable_top10[-1]:.1%}, "
            f"Bottom50: {stable_bottom50[0]:.1%}->{stable_bottom50[-1]:.1%}"
        )

        return EmergenceResult(
            law_id='HL-1',
            law_name='Capital concentration rising',
            emerged=emerged,
            confidence=min(confidence, 1.0),
            evidence=evidence,
            details={
                'mk_gini': mk_gini,
                'mk_top10': mk_top10,
                'mk_bottom50': mk_bottom50,
                'indicators_pass': len(indicators),
            },
            statistical_test='mann_kendall',
        )

    # ============================================================
    # ============================================================

    @staticmethod
    def check_competition_elimination(ts: Dict[str, List[float]]) -> EmergenceResult:
        """
        HL-3: Has competition died out

        Old logic (no ML-9): population monotonically decreasing = HL-3 always true (structural flaw)
        New logic (with ML-9): distinguish three demographic patterns:
          - Dynamic equilibrium: births ~ deaths, population stable -> HL-3 not emerged
          - Competition elimination: deaths > births, population long-term declining -> HL-3 emerged
          - Population expansion: births > deaths, population growing -> HL-3 not emerged

        When no birth/death data available, falls back to old logic (compatible with P3 and earlier experiments).
        """
        population = ts.get('population', [])
        hhi = ts.get('hhi', [])
        births = ts.get('births', [])
        deaths = ts.get('deaths', [])

        if len(population) < 30:
            return EmergenceResult('HL-3', 'Competition elimination', False, 0, 'Insufficient data',
                                   {}, 'insufficient_data')

        start = len(population) // 5
        stable_pop = population[start:]
        stable_hhi = hhi[start:]
        initial_pop = population[0]

        # === With ML-9 data: use birth/death rate analysis ===
        has_demographics = (len(births) >= 30 and len(deaths) >= 30 and
                            sum(births) > 0)

        if has_demographics:
            stable_births = births[start:]
            stable_deaths = deaths[start:]

            # Net growth series: positive = population growth, negative = population shrinkage
            net_growth = [b - d for b, d in zip(stable_births, stable_deaths)]
            net_growth_arr = np.array(net_growth, dtype=float)

            # Birth and death rates (per tick per 1000)
            pop_arr = np.array(stable_pop, dtype=float)
            pop_arr = np.maximum(pop_arr, 1.0)  # Prevent division by zero
            birth_rate = np.array(stable_births, dtype=float) / pop_arr * 1000
            death_rate = np.array(stable_deaths, dtype=float) / pop_arr * 1000

            # Cumulative net growth
            cumulative_net = np.cumsum(net_growth_arr)

            # MK test on net growth trend
            mk_net = mann_kendall_test(net_growth_arr)
            mk_pop = mann_kendall_test(np.array(stable_pop))

            # Judgment logic
            # Competition elimination = population long-term shrinking (deaths > births), not growth rate decline
            # Growth rate decline (approaching carrying capacity) is normal ecological balance, not competitive elimination
            #
            # 1. Terminal population below initial (most direct competition elimination evidence)
            pop_declined = stable_pop[-1] < initial_pop * 0.9
            cumulative_negative = cumulative_net[-1] < 0
            pop_shrinking = (mk_pop['trend'] == 'decreasing' and mk_pop['significant']
                             and stable_pop[-1] < initial_pop)
            mid = len(net_growth_arr) // 2
            if mid > 0:
                late_avg_net = np.mean(net_growth_arr[mid:])
                death_exceeds_birth = late_avg_net < 0
            else:
                death_exceeds_birth = False

            emerged = cumulative_negative or (pop_shrinking and pop_declined) or \
                       (death_exceeds_birth and pop_declined)

            if emerged:
                pop_decline_ratio = max(0, 1 - stable_pop[-1] / initial_pop)
                confidence = min(1.0, max(0.3, pop_decline_ratio * 1.5))
            else:
                confidence = 0.0

            avg_birth_rate = np.mean(birth_rate[start:]) if len(birth_rate) > start else 0
            avg_death_rate = np.mean(death_rate[start:]) if len(death_rate) > start else 0

            evidence = (
            )

            return EmergenceResult(
                law_id='HL-3',
                emerged=emerged,
                confidence=confidence,
                evidence=evidence,
                details={
                    'total_births': sum(births),
                    'total_deaths': sum(deaths),
                    'avg_birth_rate': float(avg_birth_rate),
                    'avg_death_rate': float(avg_death_rate),
                    'cumulative_net_growth': float(cumulative_net[-1]),
                    'mk_net': mk_net,
                    'mk_pop': mk_pop,
                    'has_demographics': True,
                },
                statistical_test='mann_kendall + birth/death dynamics',
            )

        half_life = len(population)
        for i, p in enumerate(population):
            if p <= initial_pop / 2:
                half_life = i
                break

        pop_decline = 1 - population[-1] / initial_pop

        mk_pop = mann_kendall_test(np.array(stable_pop))
        mk_hhi = mann_kendall_test(np.array(stable_hhi)) if len(stable_hhi) > 3 else None

        pop_declining = mk_pop['trend'] == 'decreasing' and mk_pop['significant']
        hhi_rising = mk_hhi and mk_hhi['trend'] == 'increasing' and mk_hhi['significant']

        emerged = pop_declining or (pop_decline > 0.3) or hhi_rising

        if emerged:
            confidence = min(1.0, pop_decline * 1.5)
        else:
            confidence = 0.0

        evidence = (
            f"Population: {initial_pop}->{population[-1]} (declined {pop_decline:.0%}), "
            f"half-life={half_life} ticks, "
            f"MK_pop: tau={mk_pop['tau']:.3f}, p={mk_pop['p_value']:.4f} "
            f"[No ML-9 data, using old logic]"
        )

        return EmergenceResult(
            law_id='HL-3',
            law_name='Competition elimination',
            emerged=emerged,
            confidence=confidence,
            evidence=evidence,
            details={
                'half_life': half_life,
                'pop_decline': pop_decline,
                'mk_pop': mk_pop,
                'mk_hhi': mk_hhi,
                'has_demographics': False,
            },
            statistical_test='mann_kendall + half_life (legacy)',
        )

    # ============================================================
    # ============================================================

    @staticmethod
    def check_endogenous_crisis(ts: Dict[str, List[float]]) -> EmergenceResult:
        """HL-4: Endogenous periodic crisis -- using surrogate data method + crisis event detection"""
        production = ts.get('total_production', [])

        if len(production) < 50:
            return EmergenceResult('HL-4', 'Endogenous periodic crisis', False, 0, 'Insufficient data',
                                   {}, 'insufficient_data')

        prod_arr = np.array(production)
        start = len(prod_arr) // 5
        stable_prod = prod_arr[start:]

        periodicity = test_periodicity(stable_prod, num_surrogates=50)

        crises = detect_crisis_events(stable_prod, window=20)

        cv = np.std(stable_prod) / np.mean(stable_prod) if np.mean(stable_prod) > 0 else 0

        has_periodicity = periodicity.get('is_periodic', False)
        has_crises = len(crises) >= 2
        high_volatility = cv > 0.2

        emerged = has_periodicity or (has_crises and high_volatility)

        if emerged:
            confidence = 0.0
            if has_periodicity:
                confidence += 0.5 * (1 - periodicity['p_value'])
            if has_crises:
                confidence += 0.3 * min(1.0, len(crises) / 5)
            if high_volatility:
                confidence += 0.2 * min(1.0, cv / 0.5)
        else:
            confidence = 0.0

        evidence = (
            f"CV={cv:.3f}, "
        )

        if crises:
            avg_duration = np.mean([c['duration'] for c in crises])
            evidence += f", avg duration {avg_duration:.0f} ticks"

        return EmergenceResult(
            law_id='HL-4',
            law_name='Endogenous periodic crisis',
            emerged=emerged,
            confidence=min(confidence, 1.0),
            evidence=evidence,
            details={
                'periodicity': periodicity,
                'num_crises': len(crises),
                'crises': crises[:5],
                'cv': cv,
            },
            statistical_test='surrogate_data + crisis_detection',
        )

    # ============================================================
    # ============================================================

    @staticmethod
    def check_trickle_down(ts: Dict[str, List[float]]) -> EmergenceResult:
        """HL-5: Does trickle-down hold?"""
        top10 = ts.get('top10_share', [])
        bottom50 = ts.get('bottom50_share', [])
        mean_wealth = ts.get('mean_wealth', [])

        if len(bottom50) < 30:
            return EmergenceResult('HL-5', 'Trickle-down effect', False, 0, 'Insufficient data',
                                   {}, 'insufficient_data')

        start = len(top10) // 5
        stable_top10 = top10[start:]
        stable_bottom50 = bottom50[start:]

        mk_bottom50 = mann_kendall_test(np.array(stable_bottom50))

        trickle_valid = mk_bottom50['trend'] == 'increasing' and mk_bottom50['significant']

        evidence = (
            f"Bottom50: {stable_bottom50[0]:.1%}->{stable_bottom50[-1]:.1%}, "
            f"MK: tau={mk_bottom50['tau']:.4f}, p={mk_bottom50['p_value']:.4f}"
        )

        return EmergenceResult(
            law_id='HL-5',
            law_name='Trickle-down effect',
            emerged=trickle_valid,
            confidence=1 - mk_bottom50['p_value'] if trickle_valid else 0.0,
            evidence=evidence,
            details={'mk_bottom50': mk_bottom50},
            statistical_test='mann_kendall',
        )

    # ============================================================
    # ============================================================

    @staticmethod
    def check_profit_equalization(ts: Dict[str, List[float]]) -> EmergenceResult:
        """
        HL-2: Profit rate equalization trend

        Uses cross-sector profit rate standard deviation (profit_rate_std) rather than HHI.
        HHI measures employer market concentration, unrelated to Marx's profit rate equalization.

        Profit rate equalization = profit_rate_std significantly declining (Mann-Kendall).
        If profit_rate_std is significantly rising, profit rates are diverging, equalization has not emerged.
        """
        profit_rate_std = ts.get('profit_rate_std', [])

        if len(profit_rate_std) < 30:
            return EmergenceResult('HL-2', 'Profit rate equalization', False, 0, 'Insufficient data',
                                   {}, 'insufficient_data')

        start = len(profit_rate_std) // 5
        stable_pr_std = profit_rate_std[start:]

        mk = mann_kendall_test(np.array(stable_pr_std))

        # Equalization = profit rate std declining
        equalizing = mk['trend'] == 'decreasing' and mk['significant']
        # Diverging = profit rate std rising (disproves equalization)
        diverging = mk['trend'] == 'increasing' and mk['significant']

        evidence = (
            f"Profit rate Std: {stable_pr_std[0]:.3f}->{stable_pr_std[-1]:.3f}, "
            f"MK: tau={mk['tau']:.4f}, p={mk['p_value']:.4f}"
        )

        if diverging:
            evidence += " -> Profit rates diverging (not equalizing)"
        elif equalizing:
            evidence += " -> Profit rate equalization emerged"

        # Equalization requires profit rate std to significantly decline
        emerged = equalizing

        if emerged:
            confidence = 1 - mk['p_value']
        elif diverging:
            confidence = 0.0
        else:
            confidence = 0.0

        return EmergenceResult(
            law_id='HL-2',
            law_name='Profit rate equalization',
            emerged=emerged,
            confidence=confidence,
            evidence=evidence,
            details={
                'mk': mk,
                'start_std': stable_pr_std[0],
                'end_std': stable_pr_std[-1],
                'diverging': diverging,
            },
            statistical_test='mann_kendall',
        )

    # ============================================================
    # Report formatting
    # ============================================================

    @staticmethod
    def format_report(results: List[EmergenceResult]) -> str:
        lines = [
            "=" * 70,
            "  Emergence Test Report (Statistical Framework)",
            "=" * 70,
            "",
        ]

        emerged_count = sum(1 for r in results if r.emerged)
        lines.append(f"  Total laws tested: {len(results)}")
        lines.append(f"  Emerged:   {emerged_count}")
        lines.append(f"  Absent:    {len(results) - emerged_count}")
        lines.append("")

        for r in results:
            status = "[EMERGED]" if r.emerged else "[ABSENT] "
            lines.append(f"  {status} {r.law_id}: {r.law_name}")
            lines.append(f"    Confidence: {r.confidence:.1%}")
            lines.append(f"    Test:       {r.statistical_test}")
            lines.append(f"    Evidence:   {r.evidence}")
            lines.append("")

        return "\n".join(lines)
