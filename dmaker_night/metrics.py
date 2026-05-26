"""Evaluation metrics: decision quality and interpretability."""

from functools import lru_cache

import numpy as np
from scipy.stats import spearmanr


@lru_cache(maxsize=32)
def _project_matrix(n_props, n_states):
    """Cached ``(n_props, n_states)`` projection matrix.

    ``M[idx-1, s]`` is the share of mass that proposition ``idx`` (encoded
    as a bitmask, so ``idx ∈ 1..n_props``) contributes to state ``s``.
    Multi-element subsets split their mass uniformly across their members,
    matching the original :func:`project_to_states` semantics.

    The result is cached because for a fixed ``(n_props, n_states)`` it is
    a constant table; recomputing it on every metric call dominates the
    runtime in 30-seed × 30-trial sweeps.
    """
    M = np.zeros((n_props, n_states))
    for idx in range(1, n_props + 1):
        members = [s for s in range(n_states) if (idx >> s) & 1]
        if not members:
            continue
        share = 1.0 / len(members)
        for s in members:
            M[idx - 1, s] = share
    return M


def project_to_states(pred, n_states):
    """Project a powerset BPA to a state-level probability distribution.

    Mass on a multi-element subset is split uniformly across its members.
    """
    pred = np.asarray(pred, dtype=float)
    n_props = pred.shape[0]
    M = _project_matrix(n_props, n_states)
    out = pred @ M
    s = float(np.sum(out))
    if s > 0:
        out = out / s
    return out


def accuracy(pred, true_state, n_states=None):
    """Argmax accuracy on the projected state distribution."""
    if n_states is not None and pred.shape[0] != n_states:
        pred = project_to_states(pred, n_states)
    return float(np.argmax(pred) == true_state)


def f1_score(pred, true_state, n_states=None):
    """F1 (single-label) reduces to accuracy when there is one ground-truth class."""
    return accuracy(pred, true_state, n_states=n_states)


def brier_score(pred, true_state, n_states=None):
    """Mean squared error to the one-hot ground truth."""
    if n_states is not None and pred.shape[0] != n_states:
        pred = project_to_states(pred, n_states)
    one_hot = np.zeros_like(pred)
    one_hot[true_state] = 1.0
    return float(np.mean((pred - one_hot) ** 2))


def log_likelihood(pred, true_state, n_states=None):
    """Log-likelihood of the true state."""
    if n_states is not None and pred.shape[0] != n_states:
        pred = project_to_states(pred, n_states)
    p = float(pred[true_state])
    return float(np.log(p)) if p > 0 else float("-inf")


def weight_reliability_separation(estimated, ground_truth):
    """Spearman rank correlation between estimated and ground-truth quality."""
    estimated = np.asarray(estimated, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    if np.allclose(estimated, estimated[0]) or np.allclose(ground_truth, ground_truth[0]):
        return 0.0
    corr, _ = spearmanr(estimated, ground_truth)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def reasoning_path_length(iterations):
    """Number of iterations a method takes to converge."""
    return float(iterations)


def conflict_traceability(estimated_reliabilities, conflicting_index):
    """Indicator: is the conflicting agent assigned the lowest reliability?"""
    estimated = np.asarray(estimated_reliabilities, dtype=float)
    return float(int(np.argmin(estimated)) == conflicting_index)
