"""Powerset and graph utility functions used throughout the D-MAKER framework."""

import numpy as np
from scipy.sparse.csgraph import shortest_path


def soft_bpa(observed_state, sharpness, n_states):
    """Generate a soft BPA on the singletons of an ``n_states``-element frame.

    The BPA is constructed on the powerset minus the empty set (length
    ``2 ** n_states - 1``). ``sharpness`` ∈ [0, 1] controls how
    concentrated the BPA is on the observed singleton:

        * ``sharpness ≈ 1`` puts almost all mass on ``observed_state``;
        * lower values spread mass across the other singletons,
          modelling sensor noise that produces partial support for
          several states;
        * a small residual mass on the full frame Θ is always retained,
          modelling sensor ambiguity / ignorance.

    This helper is used by all three procedural dataset generators in
    :mod:`dmaker_night.data` so they share a single canonical implementation.
    """
    n_props = 2 ** n_states - 1
    bpa = np.zeros(n_props)
    peak = sharpness
    residual_frame = 0.05 * (1 - sharpness) + 0.05  # always retain some ambiguity
    off_peak_total = max(0.0, 1.0 - peak - residual_frame)
    n_other = n_states - 1
    for s in range(n_states):
        idx = 2 ** s - 1
        if s == observed_state:
            bpa[idx] = peak
        else:
            bpa[idx] = off_peak_total / n_other if n_other > 0 else 0.0
    bpa[-1] = residual_frame
    total = bpa.sum()
    if total > 0:
        bpa = bpa / total
    return bpa


def all_pairs_shortest_path(adjacency):
    """All-pairs shortest-path distances on a binary adjacency matrix.

    Parameters
    ----------
    adjacency : (n, n) numpy.ndarray
        0/1 adjacency matrix (no self-loops required).

    Returns
    -------
    dist : (n, n) numpy.ndarray
        Pairwise shortest-path lengths (``np.inf`` if disconnected, 0 on
        the diagonal). Implemented via :func:`scipy.sparse.csgraph.shortest_path`,
        which is several orders of magnitude faster than a pure-Python
        Floyd-Warshall once ``n`` exceeds ~30.
    """
    adj = np.asarray(adjacency, dtype=float)
    return shortest_path(adj, method="auto", directed=False, unweighted=True)


def line_graph(n):
    """Adjacency matrix of an undirected line graph with n nodes."""
    adj = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def ring_graph(n):
    """Adjacency matrix of an undirected ring graph with n nodes."""
    adj = line_graph(n)
    if n > 2:
        adj[0, n - 1] = 1
        adj[n - 1, 0] = 1
    return adj


def random_graph(n, p=0.4, seed=0):
    """Adjacency matrix of an Erdős-Rényi style random graph (always connected via fallback)."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i, j] = 1
                adj[j, i] = 1
    # Ensure connectivity via a spanning chain fallback
    for i in range(n - 1):
        if not adj[i].any():
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
    return adj


def fully_connected(n):
    """Adjacency matrix of a fully connected graph (no self-loops)."""
    return np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
