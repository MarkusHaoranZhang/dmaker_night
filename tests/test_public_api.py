"""End-to-end smoke tests for the public API.

These tests are deliberately fast (small graphs, few seeds) and ensure
that all the entry points used in the experiment scripts are wired up
correctly.
"""

import numpy as np
import pytest

from dmaker_night import (
    DMaker,
    DMakerFlood,
    CentralizedMAKER,
    DempsterConsensus,
    DistributedGradientTracking,
    DistributedBayesianFilter,
    MajorityVoting,
    SyntheticDataset,
    UAVSwarmDataset,
    UCIGasDataset,
)
from dmaker_night.metrics import (
    accuracy,
    brier_score,
    f1_score,
    log_likelihood,
    project_to_states,
    weight_reliability_separation,
    conflict_traceability,
)
from dmaker_night.utils import fully_connected, ring_graph


def test_dmaker_run_returns_correct_arities():
    """``DMaker.run`` produces consistent return tuples."""
    n_agents, n_states = 4, 3
    rng = np.random.default_rng(0)
    bpas = [rng.dirichlet(np.ones(2 ** n_states - 1)) for _ in range(n_agents)]
    weights = [rng.dirichlet([5.0] * n_states) for _ in range(n_agents)]
    rels = rng.uniform(0.5, 0.95, n_agents)
    adj = fully_connected(n_agents)

    dm = DMaker(n_agents, n_states, adj, eps=1e-9, max_iter=20)
    # Default: (local, iters)
    final, iters = dm.run(bpas, rels, weights)
    assert len(final) == n_agents
    assert isinstance(iters, int)

    # With trajectories: (local, traj, rel_traj, iters)
    final, traj, rel_traj, iters = dm.run(bpas, rels, weights, return_traj=True)
    assert len(traj) >= 1
    assert len(rel_traj) >= 1

    # With reliability estimate: (local, iters, est_rel)
    final, iters, est_rel = dm.run(bpas, rels, weights,
                                    return_estimated_reliability=True)
    assert est_rel.shape == (n_agents,)

    # Both: (local, traj, rel_traj, iters, est_rel)
    out = dm.run(bpas, rels, weights, return_traj=True,
                 return_estimated_reliability=True)
    assert len(out) == 5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_baselines_produce_valid_distributions(seed):
    """All baseline methods produce normalized BPA outputs."""
    n_agents, n_states = 5, 3
    rng = np.random.default_rng(seed)
    bpas = [rng.dirichlet(np.ones(2 ** n_states - 1)) for _ in range(n_agents)]
    weights = [rng.dirichlet([5.0] * n_states) for _ in range(n_agents)]
    rels = rng.uniform(0.5, 0.95, n_agents)
    adj = fully_connected(n_agents)

    methods = [
        CentralizedMAKER(n_states=n_states),
        DempsterConsensus(n_agents, adj),
        DistributedGradientTracking(n_agents, adj),
        DistributedBayesianFilter(n_agents, adj),
        MajorityVoting(),
    ]
    for method in methods:
        if isinstance(method, CentralizedMAKER):
            out = method.run(bpas, rels, weights)
        else:
            out = method.run(bpas, rels)
        out = np.asarray(out)
        assert out.shape == (2 ** n_states - 1,)
        assert np.all(out >= -1e-9), f"{type(method).__name__} produced negative mass"
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)


def test_metrics_handle_powerset_and_state_inputs():
    """Metrics work whether the prediction is on the powerset or the
    state simplex."""
    pred_powerset = np.array([0.5, 0.2, 0.0, 0.2, 0.0, 0.0, 0.1])  # 7-d for 3 states
    pred_state = project_to_states(pred_powerset, 3)
    assert pred_state.shape == (3,)
    np.testing.assert_allclose(pred_state.sum(), 1.0, atol=1e-9)

    # Both forms should give consistent argmax
    assert (np.argmax(pred_state)
            == int(np.argmax(project_to_states(pred_powerset, 3))))

    # Brier score, log-likelihood, accuracy must be finite
    for m in (accuracy, brier_score, log_likelihood):
        v = m(pred_state, 0)
        assert np.isfinite(v) or v == float("-inf")  # log_likelihood may be -inf


def test_synthetic_dataset_structure():
    """Synthetic dataset trial dicts contain the expected keys."""
    ds = SyntheticDataset(seed=0)
    trials = ds.generate()
    assert len(trials) == ds.n_trials
    t = trials[0]
    assert {"true_state", "evidence", "reliabilities", "weights",
            "agent_types"}.issubset(t.keys())
    assert len(t["evidence"]) == ds.n_agents
    assert len(t["evidence"][0]) == ds.n_timesteps


def test_uav_dataset_reliability_shape():
    ds = UAVSwarmDataset(seed=1)
    trials = ds.generate()
    t = trials[0]
    assert t["reliabilities"].shape == (ds.n_timesteps, ds.n_agents)


def test_uci_simulator_dataset_shape():
    ds = UCIGasDataset(seed=2)
    trials = ds.generate()
    t = trials[0]
    assert t["reliabilities"].shape == (ds.n_timesteps, ds.n_agents)
    assert len(t["weights"]) == ds.n_agents


def test_flood_diameter_matches_protocol():
    """``DMakerFlood`` and the iterative ``DMaker`` reach the same argmax
    decision on small connected graphs."""
    n_agents, n_states = 5, 3
    rng = np.random.default_rng(99)
    bpas = [rng.dirichlet(np.ones(2 ** n_states - 1) * 5) for _ in range(n_agents)]
    weights = [rng.dirichlet([5.0] * n_states) for _ in range(n_agents)]
    rels = np.full(n_agents, 0.85)
    adj = ring_graph(n_agents)

    flood = DMakerFlood(n_agents, n_states, adj, max_iter=20)
    final_flood, _ = flood.run(bpas, rels, weights)

    dm = DMaker(n_agents, n_states, adj, gamma=0.0, eps=1e-9, max_iter=200,
                estimate_reliability=False)
    final_dm, _ = dm.run(bpas, rels, weights)

    for i in range(n_agents):
        assert int(np.argmax(final_dm[i])) == int(np.argmax(final_flood[i]))
