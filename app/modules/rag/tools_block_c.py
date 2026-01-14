from __future__ import annotations

import math
from typing import Literal, TypedDict


RiskLevel = Literal["High", "Moderate", "Low", "high", "normal", "low"]
Benchmark = Literal["Revenue", "PBT", "Assets", "Equity", "revenue", "profit", "assets", "equity"]


class MaterialityResult(TypedDict):
    benchmark: str
    benchmark_value: float
    risk_level: str
    is_pie: bool
    om_rate: float
    om: float
    pm_rate: float
    pm: float
    ctt_rate: float
    ctt: float
    ct: float
    rationale: str


class SampleSizeResult(TypedDict):
    method: str
    population: float
    pm: float
    confidence_level: float
    reliability_factor: float
    sampling_interval: float | None
    sample_size: int
    rationale: str


class LegalMatterResult(TypedDict):
    claim_amount: float
    probability: str
    pm: float
    is_material: bool
    disclosure_required: bool
    provision_required: bool
    fs_action: str
    is_kam: bool
    rationale: str


def calculate_materiality(
    *,
    benchmark: Benchmark,
    benchmark_value: float,
    risk_level: RiskLevel,
    is_pie: bool,
) -> MaterialityResult:
    benchmark_norm = str(benchmark).strip()
    risk_norm = str(risk_level).strip()

    # Map docs-style inputs to internal enum
    b_map = {
        "revenue": "Revenue",
        "assets": "Assets",
        "equity": "Equity",
        "profit": "PBT",
    }
    r_map = {
        "low": "Low",
        "normal": "Moderate",
        "high": "High",
    }
    benchmark_norm = b_map.get(benchmark_norm.lower(), benchmark_norm)
    risk_norm = r_map.get(risk_norm.lower(), risk_norm)

    if benchmark_norm not in {"Revenue", "PBT", "Assets", "Equity"}:
        raise ValueError("benchmark must be one of revenue/profit/assets/equity")
    if risk_norm not in {"High", "Moderate", "Low"}:
        raise ValueError("risk_level must be one of low/normal/high")

    ranges: dict[Benchmark, tuple[float, float]] = {
        "Revenue": (0.005, 0.015),
        "PBT": (0.05, 0.10),
        "Assets": (0.01, 0.02),
        "Equity": (0.02, 0.05),
    }

    pm_rates: dict[RiskLevel, float] = {
        "High": 0.55,
        "Moderate": 0.65,
        "Low": 0.75,
    }

    ctt_rates: dict[RiskLevel, float] = {
        "High": 0.03,
        "Moderate": 0.04,
        "Low": 0.05,
    }

    if benchmark_value < 0:
        raise ValueError("benchmark_value must be non-negative")

    # type ignore: we normalize benchmark_norm to internal values
    min_rate, max_rate = ranges[benchmark_norm]  # type: ignore[index]

    percentile = 0.5
    if risk_norm == "High":
        percentile = 0.25
    elif risk_norm == "Low":
        percentile = 0.75

    if is_pie:
        percentile = max(0.10, percentile - 0.15)

    om_rate = min_rate + (max_rate - min_rate) * percentile
    om = float(benchmark_value) * float(om_rate)

    pm_rate = pm_rates[risk_norm]  # type: ignore[index]
    pm = om * pm_rate

    ctt_rate = ctt_rates[risk_norm]  # type: ignore[index]
    ctt = om * ctt_rate

    rationale = (
        f"OM={benchmark_value:,.0f}×{om_rate:.2%} ({benchmark_norm}); "
        f"PM={pm_rate:.0%} of OM (risk={risk_norm}); "
        f"CTT={ctt_rate:.0%} of OM; PIE={is_pie}"
    )

    return {
        "benchmark": benchmark_norm,
        "benchmark_value": float(benchmark_value),
        "risk_level": risk_norm,
        "is_pie": bool(is_pie),
        "om_rate": float(om_rate),
        "om": float(round(om, 2)),
        "pm_rate": float(pm_rate),
        "pm": float(round(pm, 2)),
        "ctt_rate": float(ctt_rate),
        "ctt": float(round(ctt, 2)),
        "ct": float(round(ctt, 2)),
        "rationale": rationale,
    }


def calculate_sample_size(
    *,
    population: float,
    pm: float,
    confidence_level: float = 0.95,
    expected_errors: int = 0,
) -> SampleSizeResult:
    if population < 0:
        raise ValueError("population must be non-negative")
    if pm <= 0:
        raise ValueError("pm must be positive")

    reliability_factor_map = {
        0.90: 2.31,
        0.95: 3.0,
        0.99: 4.61,
    }

    rf = reliability_factor_map.get(round(float(confidence_level), 2), 3.0)

    sampling_interval: float | None = None

    if population >= 100000 or population >= pm * 2:
        sampling_interval = pm / rf
        sample_size = int(math.ceil(population / sampling_interval)) if sampling_interval > 0 else 0
        sample_size = max(1, sample_size)
        method = "MUS"
        rationale = f"MUS: SI=PM/RF={pm:,.0f}/{rf:.2f}={sampling_interval:,.0f}; n=TBV/SI"
    else:
        n = int(population)
        base = 10
        if n >= 1000:
            base = 40
        elif n >= 200:
            base = 25

        bump = 0
        if expected_errors > 0:
            bump += min(10, int(expected_errors))
        if confidence_level >= 0.99:
            bump += int(math.ceil(base * 0.25))

        sample_size = min(n, base + bump)
        sample_size = max(1, sample_size)
        method = "NonStat_Baseline"
        rationale = "Baseline sample size heuristic (C4 matrix) because population looks count-based"

    return {
        "method": method,
        "population": float(population),
        "pm": float(pm),
        "confidence_level": float(confidence_level),
        "reliability_factor": float(rf),
        "sampling_interval": float(round(sampling_interval, 2)) if sampling_interval is not None else None,
        "sample_size": int(sample_size),
        "rationale": rationale,
    }


def assess_legal_matter(
    *,
    claim_amount: float,
    probability: Literal["probable", "possible", "remote"],
    pm: float,
    outcome_estimable: bool = True,
) -> LegalMatterResult:
    if claim_amount < 0:
        raise ValueError("claim_amount must be non-negative")
    if pm <= 0:
        raise ValueError("pm must be positive")

    is_material = claim_amount >= pm

    provision_required = False
    disclosure_required = False
    fs_action = "None"

    if probability == "remote":
        provision_required = False
        disclosure_required = False
        fs_action = "None"
        rationale = "IAS 37: remote probability"
    elif probability == "possible":
        provision_required = False
        disclosure_required = True
        fs_action = "Contingent Liability Disclosure"
        rationale = "IAS 37: possible -> contingent liability disclosure"
    else:
        if outcome_estimable:
            provision_required = True
            disclosure_required = True
            fs_action = "Provision"
            rationale = "IAS 37: probable + estimable -> provision"
        else:
            provision_required = False
            disclosure_required = True
            fs_action = "Contingent Liability Disclosure"
            rationale = "IAS 37: probable + not estimable -> disclose range"

    is_kam = bool(is_material and probability in {"probable", "possible"} and claim_amount >= 2 * pm)
    if is_kam:
        rationale = f"{rationale}; KAM candidate (>=2x PM)"

    return {
        "claim_amount": float(claim_amount),
        "probability": probability,
        "pm": float(pm),
        "is_material": bool(is_material),
        "disclosure_required": bool(disclosure_required),
        "provision_required": bool(provision_required),
        "fs_action": fs_action,
        "is_kam": bool(is_kam),
        "rationale": rationale,
    }
