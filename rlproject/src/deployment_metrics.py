import numpy as np


def _as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def compute_fixed_horizon_tiz(rows, duration_target_s, done_reason):
    """Integrate in-ZPD time and normalize it by the configured horizon."""
    horizon_s = _as_float(duration_target_s)
    if horizon_s is None or horizon_s <= 0.0:
        raise ValueError("duration_target_s must be positive")

    samples = []
    for row in rows:
        t_task_s = _as_float(row.get("t_task_s"))
        if t_task_s is None:
            continue
        samples.append((float(np.clip(t_task_s, 0.0, horizon_s)), _as_bool(row.get("in_zpd"))))

    if not samples:
        return {
            "zpd_time_s": 0.0,
            "tiz_fixed_horizon_fraction": 0.0,
        }

    samples.sort(key=lambda sample: sample[0])
    terminal_s = horizon_s if str(done_reason) == "timeout" else min(samples[-1][0], horizon_s)
    if terminal_s <= 0.0:
        return {
            "zpd_time_s": 0.0,
            "tiz_fixed_horizon_fraction": 0.0,
        }

    zpd_time_s = float(samples[0][1]) * min(samples[0][0], terminal_s)
    for index, (start_s, in_zpd) in enumerate(samples):
        if start_s >= terminal_s:
            break
        end_s = terminal_s
        if index + 1 < len(samples):
            end_s = min(samples[index + 1][0], terminal_s)
        if end_s > start_s and in_zpd:
            zpd_time_s += end_s - start_s

    zpd_time_s = float(np.clip(zpd_time_s, 0.0, horizon_s))
    return {
        "zpd_time_s": zpd_time_s,
        "tiz_fixed_horizon_fraction": zpd_time_s / horizon_s,
    }
