"""utils.py

Shared mathematical utilities used by handle_data.py and plot_results.py.

Keeping these here avoids circular imports: both modules need `smooth` and
the convergence helpers, but neither should depend on the other.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth(values: list[float], window: int = 11) -> np.ndarray:
    """Centered moving average with truncated windows at the edges.

    For each position i, averages over the largest centered window that fits
    within the array bounds, up to the given window size. Boundary points use
    a smaller window rather than being padded with zeros.

    For even-sized windows, floor(window/2) points are taken to the left and
    (window - floor(window/2) - 1) to the right, so interior points always
    average exactly `window` values.

    Example:
        smooth([1, 2, 3, 4, 5], window=3) -> [1.5, 2.0, 3.0, 4.0, 4.5]
          - i=0: avg([1, 2])    = 1.5  (only 1 right neighbour available)
          - i=1: avg([1, 2, 3]) = 2.0  (full window)
          - i=2: avg([2, 3, 4]) = 3.0  (full window)
          - i=3: avg([3, 4, 5]) = 4.0  (full window)
          - i=4: avg([4, 5])    = 4.5  (only 1 left neighbour available)

    Args:
        values: Sequence of scalar values to smooth.
        window: Total window size. Should be a positive odd integer for
                symmetric smoothing; even values are supported but produce a
                slightly left-biased window.

    Returns:
        np.ndarray of the same length as values.
    """
    arr: np.ndarray = np.array(values, dtype=float)
    n: int          = len(arr)
    half_left: int  = window // 2
    half_right: int = window - half_left - 1
    smoothed: np.ndarray = np.empty(n)
    for i in range(n):
        start: int  = max(0, i - half_left)
        end: int    = min(n - 1, i + half_right)
        smoothed[i] = arr[start : end + 1].mean()
    return smoothed


# ---------------------------------------------------------------------------
# Relative-change convergence helpers
# ---------------------------------------------------------------------------

def compute_relative_change(values: list[float],
                             epsilon: float = 1e-8) -> np.ndarray:
    """Pointwise relative change between consecutive smoothed values.

    Smooths `values` first, then computes:

        rel_change[i] = |smoothed[i+1] - smoothed[i]| / (|smoothed[i]| + epsilon)

    for i in 0 .. len(values)-2.  The epsilon prevents division by zero when a
    metric flattens at or near zero (e.g. a loss that has converged to zero).

    Args:
        values:  Sequence of scalar metric values.
        epsilon: Small constant added to the denominator.

    Returns:
        np.ndarray of length len(values) - 1.  Empty array if len(values) < 2.
    """
    if len(values) < 2:
        return np.array([], dtype=float)
    smoothed: np.ndarray = smooth(values)
    diffs: np.ndarray    = np.abs(np.diff(smoothed))
    denoms: np.ndarray   = np.abs(smoothed[:-1]) + epsilon
    return diffs / denoms


def first_convergence_index(rel_changes: np.ndarray,
                             threshold: float) -> int | None:
    """Return the first index (into the original values array) after which the
    convergence criterion holds permanently.

    Uses the rfind approach: finds the *last* index where ``rel_changes``
    violates the criterion (i.e. is >= threshold), then returns the next index.
    This guarantees the returned point marks the start of a permanently-
    converged tail rather than the first transient dip below the threshold.

    The returned index maps back to the original values array: if
    ``rel_changes`` has length n-1 (computed from a values series of length n),
    the valid return range is 0 .. n-2.

    Args:
        rel_changes: Array of relative changes, length n-1, from
                     compute_relative_change().
        threshold:   Convergence threshold (e.g. 0.15 for 15 % relative change).

    Returns:
        Integer index into the original values array, or None if:
          - rel_changes is empty (fewer than 2 data points), or
          - the criterion is never met (every value >= threshold), or
          - the last violation is the final element of rel_changes (no
            confirmed tail of stable values remains).
    """
    if len(rel_changes) == 0:
        return None

    violations: np.ndarray = np.where(rel_changes >= threshold)[0]

    if len(violations) == 0:
        # All transitions are already below threshold from the start.
        return 0

    last_violation_idx: int = int(violations[-1])
    convergence_idx: int    = last_violation_idx + 1

    # rel_changes has length n-1 (indices 0 .. n-2).
    # convergence_idx == n-1 means the last transition was a violation — no
    # confirmed stable tail exists beyond the end of the data.
    if convergence_idx >= len(rel_changes):
        return None

    return convergence_idx
