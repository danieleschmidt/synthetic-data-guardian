"""Statistical validation for synthetic data."""
import math
from typing import List, Dict, Any, Tuple


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def validate_column_distribution(
    col: List[float], expected_min: float, expected_max: float, tolerance: float = 0.15
) -> Dict[str, Any]:
    """Check if column values fall within expected range with tolerance."""
    actual_min = min(col)
    actual_max = max(col)
    actual_mean = mean(col)
    actual_std = std(col)

    range_size = expected_max - expected_min
    min_ok = actual_min >= expected_min - range_size * tolerance
    max_ok = actual_max <= expected_max + range_size * tolerance

    return {
        "valid": min_ok and max_ok,
        "actual_min": actual_min,
        "actual_max": actual_max,
        "actual_mean": actual_mean,
        "actual_std": actual_std,
        "min_ok": min_ok,
        "max_ok": max_ok,
    }


def correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = math.sqrt(
        sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)
    )
    return num / denom if denom != 0 else 0.0


def validate_correlation_preserved(
    orig_x: List[float],
    orig_y: List[float],
    syn_x: List[float],
    syn_y: List[float],
    tolerance: float = 0.3,
) -> Dict[str, Any]:
    """Check that correlation between two columns is preserved within tolerance."""
    orig_corr = correlation(orig_x, orig_y)
    syn_corr = correlation(syn_x, syn_y)
    preserved = abs(orig_corr - syn_corr) <= tolerance
    return {
        "preserved": preserved,
        "original_correlation": orig_corr,
        "synthetic_correlation": syn_corr,
        "delta": abs(orig_corr - syn_corr),
    }


def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run full validation suite on a dataset."""
    if not records:
        return {"valid": False, "error": "Empty dataset"}

    results = {"valid": True, "checks": {}}

    # Validate age column if present
    if "age" in records[0]:
        ages = [r["age"] for r in records]
        age_check = validate_column_distribution(ages, 0, 120)
        results["checks"]["age"] = age_check
        if not age_check["valid"]:
            results["valid"] = False

    # Validate salary column if present
    if "salary" in records[0]:
        salaries = [r["salary"] for r in records]
        salary_check = validate_column_distribution(salaries, 0, 1_000_000)
        results["checks"]["salary"] = salary_check
        if not salary_check["valid"]:
            results["valid"] = False

    results["record_count"] = len(records)
    return results
