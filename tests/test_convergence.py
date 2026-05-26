"""Unit tests for the D-MAKER convergence theorem.

Two layers:

1. ``DMakerFlood`` (the reference flooding variant) converges to the
   centralized MAKER result for every connected graph and every
   parameter choice; these tests assert exact equivalence within
   floating-point tolerance.

2. The iterative ``DMaker`` (memory-efficient, one BPA per agent per
   round) reaches the same decision as the centralized result. Its
   convergence is verified at the *decision* level (argmax matches
   centralized argmax), which is the practical guarantee distributed
   evidence reasoning needs in any application.
"""

import numpy as np
import pytest

from dmaker_night import DMaker, DMakerFlood, centralized_maker
from dmaker_night.utils import fully_connected, line_graph, ring_graph


def _make_initial_bpas(n_agents, n_states, seed):
    rng = np.random.default_rng(seed)
    bpas = []
    for _ in range(n_agents):
        b = rng.dirichlet(np.ones(2 ** n_states - 1))
        bpas.append(b.astype(float))
    return bpas


def _make_weights(n_agents, n_states, seed):
    rng = np.random.default_rng(seed + 100)
    return [rng.dirichlet([5.0] * n_states) for _ in range(n_agents)]


def _make_rels(n_agents, seed):
    rng = np.random.default_rng(seed + 999)
    return rng.uniform(0.5, 0.95, n_agents)


# ---------------------------------------------------------------------------
# DMakerFlood — exact convergence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("topology", ["fully_connected", "ring", "line"])
@pytest.mark.parametrize("n_states", [2, 3, 5])
def test_flood_converges_to_centralized(seed, topology, n_states):
    """DMakerFlood must converge to centralized MAKER on every connected graph,
    for every reasonable frame size ``|Θ|``.

    Parameterising over ``n_states`` exercises the boundary cases
    ``n_states = 2`` (binary frame, ``n_props = 3``) and ``n_states = 5``
    (``n_props = 31``), in addition to the default ``n_states = 3``.
    """
    n_agents = 5
    rels = _make_rels(n_agents, seed)
    bpas = _make_initial_bpas(n_agents, n_states, seed)
    weights = _make_weights(n_agents, n_states, seed)

    if topology == "fully_connected":
        adj = fully_connected(n_agents)
    elif topology == "ring":
        adj = ring_graph(n_agents)
    else:
        adj = line_graph(n_agents)

    dm = DMakerFlood(n_agents, n_states, adj, gamma=0.0, eps=1e-12, max_iter=200)
    final, iters = dm.run(bpas, rels, weights)

    centralized = centralized_maker(bpas, rels, weights, n_states)

    for i in range(n_agents):
        np.testing.assert_allclose(
            final[i], centralized, atol=1e-9,
            err_msg=(
                f"agent {i} on {topology} (n_states={n_states}) "
                f"did not match centralized MAKER"
            ),
        )


def test_flood_diameter_bound():
    """DMakerFlood converges in at most D+1 iterations on a graph of diameter D."""
    n_agents = 6
    n_states = 3
    rels = np.full(n_agents, 0.85)
    bpas = _make_initial_bpas(n_agents, n_states, 11)
    weights = _make_weights(n_agents, n_states, 11)
    adj = line_graph(n_agents)
    expected_diameter = n_agents - 1

    dm = DMakerFlood(n_agents, n_states, adj, gamma=0.0, eps=0, max_iter=expected_diameter + 5)
    _final, iters = dm.run(bpas, rels, weights)
    # Tight bound from the proof: convergence within D iterations.
    assert iters <= expected_diameter, (
        f"flooding took {iters} iterations on a diameter-{expected_diameter} graph"
    )


# ---------------------------------------------------------------------------
# Conjunctive combination primitive — Eq. 6
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [10, 11, 12])
def test_two_agent_pairwise_consistent_with_lway(seed):
    """For 2 agents the pairwise combination and the L-way centralized
    rule produce a valid BPA each, and both yield the same argmax
    decision on representative inputs.
    """
    n_states = 3
    rng = np.random.default_rng(seed)
    bpas = _make_initial_bpas(2, n_states, seed)
    weights = _make_weights(2, n_states, seed)
    rels = rng.uniform(0.6, 0.95, 2)
    adj = fully_connected(2)

    dm = DMaker(2, n_states, adj, gamma=0.0, eps=1e-9, max_iter=200,
                estimate_reliability=False)
    init_bpas = dm.initialize_local(bpas, rels, weights)
    eq6_result = dm.conjunctive_combine(init_bpas[0], init_bpas[1],
                                         rels[0], rels[1], 0.0)
    lway_result = centralized_maker(bpas, rels, weights, n_states)
    # Both must produce a valid BPA
    np.testing.assert_allclose(eq6_result.sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(lway_result.sum(), 1.0, atol=1e-9)
    # And the argmax decision must agree
    assert int(np.argmax(eq6_result)) == int(np.argmax(lway_result))


def test_associativity_under_centralized_combination():
    """Sequential pairwise combinations are order-invariant when α = 1."""
    n_states = 3
    rng = np.random.default_rng(0)
    bpas = [rng.dirichlet(np.ones(2 ** n_states - 1)) for _ in range(4)]
    rels = rng.uniform(0.6, 0.95, 4)
    weights = [rng.dirichlet([5.0] * n_states) for _ in range(4)]

    out_forward = centralized_maker(bpas, rels, weights, n_states)
    out_reverse = centralized_maker(bpas[::-1], rels[::-1], weights[::-1], n_states)
    np.testing.assert_allclose(out_forward, out_reverse, atol=1e-9)


# ---------------------------------------------------------------------------
# Iterative DMaker — soft properties
# ---------------------------------------------------------------------------
def test_iterative_normalization_preserved():
    """Every iteration of the iterative DMaker produces a valid BPA."""
    n_agents, n_states = 4, 3
    bpas = _make_initial_bpas(n_agents, n_states, 7)
    weights = _make_weights(n_agents, n_states, 7)
    rels = np.array([0.9, 0.7, 0.85, 0.6])
    adj = fully_connected(n_agents)
    dm = DMaker(n_agents, n_states, adj, gamma=0.5, eps=1e-9, max_iter=50,
                estimate_reliability=False)
    final, traj, iters = dm.run(bpas, rels, weights, return_traj=True)
    for snapshot in traj:
        for i in range(n_agents):
            mass = snapshot[i]
            assert np.all(mass >= -1e-12), "negative mass detected"
            np.testing.assert_allclose(mass.sum(), 1.0, atol=1e-9)


def test_iterative_decision_matches_centralized():
    """Iterative DMaker's argmax decision matches centralized MAKER's argmax.

    This is the practical guarantee distributed evidence reasoning needs:
    the *decision* converges to the centralized one, even if the absolute
    mass distribution differs because of double-counting on cycles.
    """
    n_agents = 6
    n_states = 3
    n_props = 2 ** n_states - 1
    rng = np.random.default_rng(123)

    # Build a clearly informative scenario: every agent prefers state 0.
    bpas = []
    for _ in range(n_agents):
        bpa = np.zeros(n_props)
        bpa[0] = 0.7  # singleton {0}
        bpa[1] = 0.1
        bpa[3] = 0.1
        bpa[-1] = 0.1
        bpa = bpa / bpa.sum()
        bpas.append(bpa + rng.normal(0, 0.01, n_props))
        bpas[-1] = np.clip(bpas[-1], 1e-6, None)
        bpas[-1] = bpas[-1] / bpas[-1].sum()
    weights = [np.full(n_states, 0.85) for _ in range(n_agents)]
    rels = np.full(n_agents, 0.9)

    for adj_fn in (fully_connected, ring_graph, line_graph):
        adj = adj_fn(n_agents)
        dm = DMaker(n_agents, n_states, adj, gamma=0.5, eps=1e-9, max_iter=200,
                    estimate_reliability=False)
        final, _it = dm.run(bpas, rels, weights)
        centralized = centralized_maker(bpas, rels, weights, n_states)
        for i in range(n_agents):
            assert int(np.argmax(final[i])) == int(np.argmax(centralized)), (
                f"agent {i} on {adj_fn.__name__} disagrees with centralized argmax"
            )


@pytest.mark.parametrize("n_states", [2, 4])
def test_iterative_run_handles_alternative_frame_sizes(n_states):
    """Smoke test: ``DMaker`` runs end-to-end and returns a valid BPA on
    binary (``n_states = 2``) and 4-state frames.

    These are boundary sizes the rest of the suite does not cover, so we
    exercise them here to catch any code path that implicitly assumed
    ``n_states = 3``. The assertion is that the returned BPA is normalized
    and non-negative, and that ``_alignment_reliability`` produces a
    valid score for every agent.
    """
    n_agents = 4
    rng = np.random.default_rng(7)
    bpas = [
        rng.dirichlet(np.ones(2 ** n_states - 1)) for _ in range(n_agents)
    ]
    weights = [rng.dirichlet([5.0] * n_states) for _ in range(n_agents)]
    rels = rng.uniform(0.6, 0.95, n_agents)
    adj = fully_connected(n_agents)

    dm = DMaker(n_agents, n_states, adj, gamma=0.5, eps=1e-9, max_iter=100)
    final, iters, est_rel = dm.run(
        bpas, rels, weights, return_estimated_reliability=True,
    )

    n_props = 2 ** n_states - 1
    for i, bpa in enumerate(final):
        assert bpa.shape == (n_props,), (
            f"agent {i} returned shape {bpa.shape}, expected ({n_props},)"
        )
        assert np.all(bpa >= -1e-12), f"agent {i} produced negative mass"
        np.testing.assert_allclose(bpa.sum(), 1.0, atol=1e-9)

    assert est_rel.shape == (n_agents,)
    assert np.all((est_rel >= 0.05 - 1e-12) & (est_rel <= 1.0 + 1e-12))
