import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import ttest_1samp, ttest_rel
from statsmodels.stats.multitest import fdrcorrection


def focus_on_decision_subspace(
    subject_patterns: np.ndarray,
    t_baseline: int,
    alpha: float = 0.05,
    correction: str | None = None,
    return_indices: bool = False,
):
    """
    Standalone helper to select sensors ("decision subspace") and return patterns restricted
    to those sensors. Average patterns per subject are compared before and after the baseline
    with paired t-test across subjects. Only sensors with significant difference are kept.

    Parameters
    ----------
    subject_patterns : np.ndarray
        3D array of shape (n_subjects, n_times, n_sensors).
    t_baseline : int
        Number of timepoints at beginning to use as baseline.
    alpha : float
        Significance threshold.
    correction : {None, "fdr", "bonferroni"}
        Multiple-comparison correction across sensors.
    return_indices : bool
        If True, also return selected sensor indices (and corrected p-values).

    Returns
    -------
    focused_patterns : np.ndarray
        Array of shape (n_subjects, n_times, n_selected_sensors).
    (optional) selected_idx : np.ndarray
        1D int array of selected sensor indices.
    (optional) pvals_corr : np.ndarray
        1D float array of p-values (corrected if correction is not None).
    """
    x = np.asarray(subject_patterns)
    if x.ndim != 3:
        raise ValueError(
            f"`subject_patterns` must be 3D (subjects, times, sensors), got shape {x.shape}."
        )

    n_subjects, n_times, n_sensors = x.shape
    if n_subjects < 2:
        raise ValueError("Paired t-test across subjects requires at least 2 subjects.")
    if not isinstance(t_baseline, (int, np.integer)):
        raise TypeError(
            f"`t_baseline` must be an int, got {type(t_baseline).__name__}."
        )
    if not (1 <= t_baseline < n_times):
        raise ValueError(
            f"`t_baseline` must satisfy 1 <= t_baseline < n_times. Got {t_baseline}, n_times={n_times}."
        )

    # --- Selection rule (as given) ---
    baseline_avg = x[:, :t_baseline, :].mean(axis=1)  # (subjects, sensors)
    sensor_avgs = x[:, t_baseline::, :].mean(axis=1)  # (subjects, sensors)

    # Paired t-test across subjects
    t_vals, pvals = ttest_rel(sensor_avgs, baseline_avg, axis=0, nan_policy="omit")
    # Replace NaNs with non-significant p=1.0
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)

    # --- Multiple comparisons ---
    if correction is None:
        pvals_corr = pvals
        passed = pvals_corr <= alpha

    elif correction.lower() == "bonferroni":
        pvals_corr = pvals  # keep raw p-values; threshold changes
        passed = pvals <= (alpha / n_sensors)

    elif correction.lower() == "fdr":
        passed, pvals_corr = fdrcorrection(pvals, alpha=alpha)

    else:
        raise ValueError("`correction` must be one of: None, 'fdr', 'bonferroni'.")

    selected_idx = np.where(passed)[0]

    if selected_idx.size == 0:
        # warn and return empty array with correct shape
        print(
            "No sensors selected (empty decision subspace). "
            "Try alpha=0.1, correction=None, or verify patterns differ from baseline."
        )
        focused = x[..., :0]
        return (focused, selected_idx, pvals_corr) if return_indices else focused

    focused = x[..., selected_idx]
    return (focused, selected_idx, pvals_corr) if return_indices else focused


def compute_feature_density(
    subject_patterns: np.ndarray,
    alpha: float = 0.05,
    correction: str | None = None,
    return_mask: bool = False,
    return_pvals: bool = False,
):
    """
    Compute feature density over time = proportion of significant sensors at each timepoint,

    Parameters
    ----------
    subject_patterns : np.ndarray
        3D array of shape (n_subjects, n_times, n_sensors).
    alpha : float
        Significance threshold.
    correction : {None, "fdr", "bonferroni"}
        Multiple-comparison correction across sensors at each timepoint.
    return_mask : bool
        If True, also return a boolean array (n_times, n_sensors) indicating significance.
    return_pvals : bool
        If True, also return p-values (corrected if correction == "fdr"; otherwise raw).

    Returns
    -------
    density : np.ndarray
        Shape (n_times,), fraction of significant sensors at each timepoint.
    (optional) sig_mask : np.ndarray
        Shape (n_times, n_sensors), boolean significance mask.
    (optional) p_values : np.ndarray
        Shape (n_times, n_sensors), p-values (raw or FDR-adjusted as noted above).
    """
    x = np.asarray(subject_patterns)
    if x.ndim != 3:
        raise ValueError(
            f"`subject_patterns` must be 3D (subjects, times, sensors), got shape {x.shape}."
        )

    n_subjects, n_times, n_sensors = x.shape
    if n_subjects < 2:
        raise ValueError(
            "One-sample t-test across subjects requires at least 2 subjects."
        )

    # p-values per timepoint & sensor
    p_values = np.empty((n_times, n_sensors), dtype=float)
    sig_mask = np.zeros((n_times, n_sensors), dtype=bool)

    for t in range(n_times):
        _, pvals = ttest_1samp(x[:, t, :], popmean=0.0, axis=0, nan_policy="omit")
        pvals = np.where(np.isfinite(pvals), pvals, 1.0)

        if correction is None:
            passed = pvals <= alpha
            p_values[t] = pvals

        else:
            c = correction.lower()
            if c == "bonferroni":
                passed = pvals <= (alpha / n_sensors)
                p_values[t] = pvals  # keep raw pvals; threshold changes

            elif c == "fdr":
                passed, pvals_adj = fdrcorrection(pvals, alpha=alpha)
                p_values[t] = pvals_adj

            else:
                raise ValueError(
                    "`correction` must be one of: None, 'fdr', 'bonferroni'."
                )

        sig_mask[t] = passed

    density = sig_mask.mean(axis=1)  # proportion significant sensors per timepoint

    outputs = [density]
    if return_mask:
        outputs.append(sig_mask)
    if return_pvals:
        outputs.append(p_values)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def get_rotation_angle_matrix(patterns):
    """
    Compute cosine distance between each pair of patterns,
    Parameters:
    - patterns: array of shape (n_times, n_sensors)
    Returns:
    - angle_matrix: rotation angles between each pair of patterns
    """
    if patterns.ndim == 2:
        n_times, n_sensors = patterns.shape
        n_classes = 2
    else:
        raise ValueError("patterns with more than 2 dimensions not supported yet.")

    angle_matrix = np.zeros((n_times, n_times))
    for t1 in range(n_times):
        w1 = patterns[t1]
        for t2 in range(n_times):
            w2 = patterns[t2]
            cos_dist = cosine(w1, w2)
            angle = np.arccos(1 - cos_dist)
            angle_matrix[t1, t2] = angle

    return angle_matrix
