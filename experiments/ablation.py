"""Ablation study (Section 5.3): isolate the contribution of each D-MAKER component.

Each component variant is evaluated against the full D-MAKER on the synthetic
high-conflict scenario. Results are averaged over ``N_SEEDS`` independent
random seeds and significance is assessed with the Wilcoxon signed-rank test.
"""

import numpy as np
from scipy.stats import wilcoxon

from dmaker_night import DMaker, SyntheticDataset
from dmaker_night.metrics import (
    accuracy,
    brier_score,
    conflict_traceability,
    log_likelihood,
    project_to_states,
    reasoning_path_length,
)
from dmaker_night.utils import fully_connected
from ._common import N_SEEDS, tables_dir


def _final_distribution(final_local, n_states):
    avg = np.mean(np.array(final_local), axis=0)
    return project_to_states(avg, n_states)


def _no_residual_run(dm, initial_bpas, reliabilities, weights, n_states):
    """Variant that places residual mass on a singleton instead of the full frame.

    We re-implement just the initialization step here and reuse the iterative
    combination from ``DMaker``.
    """
    n_props = dm.n_props
    local = []
    for i in range(dm.n_agents):
        w = np.asarray(weights[i])
        p = np.asarray(initial_bpas[i])
        weighted = np.zeros(n_props)
        for state in range(n_states):
            idx = 2 ** state - 1
            weighted[idx] = w[state] * p[idx]
        denom = float(np.sum(weighted) + (1.0 - reliabilities[i]))
        omega_i = 1.0 / denom if denom > 0 else 1.0
        m = omega_i * weighted
        # Residual mass goes to the most-supported singleton instead of 2^Theta
        top_idx = int(np.argmax(m))
        m[top_idx] += omega_i * (1.0 - reliabilities[i])
        s = float(np.sum(m))
        if s > 0:
            m = m / s
        local.append(m)

    iters = 0
    rel_curr = np.asarray(reliabilities, dtype=float).copy()
    for k in range(1, dm.max_iter):
        new_local = []
        for i in range(dm.n_agents):
            neighbors = np.where(dm.adj[i] == 1)[0]
            combined = local[i].copy()
            for j in neighbors:
                combined = dm.conjunctive_combine(
                    combined, local[j],
                    rel_curr[i], rel_curr[j],
                    dm.dist[i, j],
                )
            new_local.append(combined)
        max_change = max(
            float(np.max(np.abs(new_local[i] - local[i])))
            for i in range(dm.n_agents)
        )
        local = new_local
        iters = k
        if max_change < dm.eps:
            break
    return local, iters


def _algebraic_average(initial_bpas, n_states):
    avg = np.mean([np.asarray(b, dtype=float) for b in initial_bpas], axis=0)
    return project_to_states(avg, n_states)


def _evaluate_seed(seed):
    """Run all variants on one synthetic dataset realization.

    Returns a dict of per-method per-trial metric arrays.

    Each (trial, time-step) pair is treated as an independent
    observation, matching the paper's "50 independent trials, each
    with 10 time steps" specification (Section 5.1.1).
    """
    dataset = SyntheticDataset(n_agents=8, n_states=3, n_trials=50, n_timesteps=10, seed=seed)
    trials = dataset.generate()
    n_agents = dataset.n_agents
    n_states = dataset.n_states
    adj = fully_connected(n_agents)
    conflicting_idx = dataset.conflicting_index

    methods = [
        "Full D-MAKER",
        "w/o reliability distinction",
        "w/o dependence modeling",
        "w/o residual support",
        "Algebraic averaging",
    ]
    metrics = {m: {"acc": [], "brier": [], "ll": [], "iters": [], "trace": []} for m in methods}

    # Expand each trial's time-step axis into independent observations.
    expanded = []
    for trial in trials:
        evidence = trial["evidence"]
        rel = np.asarray(trial["reliabilities"])
        weights = trial["weights"]
        n_t = len(evidence[0])
        for t in range(n_t):
            bpas_t = [np.asarray(evidence[a][t], dtype=float) for a in range(n_agents)]
            rel_t = rel[t] if rel.ndim == 2 else rel
            expanded.append({
                "true_state": trial["true_state"],
                "bpas": bpas_t,
                "rels": np.asarray(rel_t, dtype=float),
                "weights": weights,
            })

    for trial in expanded:
        true_state = trial["true_state"]
        bpas = trial["bpas"]
        rels = trial["rels"]
        weights = trial["weights"]

        # Full D-MAKER
        dm = DMaker(n_agents, n_states, adj, gamma=0.5)
        final, iters, est_rel = dm.run(bpas, rels, weights, return_estimated_reliability=True)
        dist = _final_distribution(final, n_states)
        metrics["Full D-MAKER"]["acc"].append(accuracy(dist, true_state))
        metrics["Full D-MAKER"]["brier"].append(brier_score(dist, true_state))
        metrics["Full D-MAKER"]["ll"].append(log_likelihood(dist, true_state))
        metrics["Full D-MAKER"]["iters"].append(reasoning_path_length(iters))
        metrics["Full D-MAKER"]["trace"].append(conflict_traceability(est_rel, conflicting_idx))

        # w/o reliability distinction: removes the per-agent reliability
        # parameter entirely (every agent declared r = 0.999) AND the
        # alignment-based reliability trace (which depends on the
        # framework distinguishing trustworthy from untrustworthy
        # agents). This is the strongest ablation: with no notion of
        # reliability whatsoever, the conflict agent cannot be traced
        # and the conjunctive rule treats all evidence as equally
        # informative.
        dm_nr = DMaker(n_agents, n_states, adj, gamma=0.5,
                       estimate_reliability=False)
        final_nr, iters_nr = dm_nr.run(
            bpas, np.ones(n_agents) * 0.999, weights,
        )
        dist_nr = _final_distribution(final_nr, n_states)
        metrics["w/o reliability distinction"]["acc"].append(accuracy(dist_nr, true_state))
        metrics["w/o reliability distinction"]["brier"].append(brier_score(dist_nr, true_state))
        metrics["w/o reliability distinction"]["ll"].append(log_likelihood(dist_nr, true_state))
        metrics["w/o reliability distinction"]["iters"].append(reasoning_path_length(iters_nr))
        # No reliability distinction means no per-agent trace: by
        # construction the variant returns identical reliabilities for
        # every agent, so traceability is structurally zero.
        metrics["w/o reliability distinction"]["trace"].append(0.0)

        # w/o dependence modeling (gamma = 0)
        dm_nd = DMaker(n_agents, n_states, adj, gamma=0.0)
        final_nd, iters_nd, est_rel_nd = dm_nd.run(
            bpas, rels, weights, return_estimated_reliability=True,
        )
        dist_nd = _final_distribution(final_nd, n_states)
        metrics["w/o dependence modeling"]["acc"].append(accuracy(dist_nd, true_state))
        metrics["w/o dependence modeling"]["brier"].append(brier_score(dist_nd, true_state))
        metrics["w/o dependence modeling"]["ll"].append(log_likelihood(dist_nd, true_state))
        metrics["w/o dependence modeling"]["iters"].append(reasoning_path_length(iters_nd))
        metrics["w/o dependence modeling"]["trace"].append(
            conflict_traceability(est_rel_nd, conflicting_idx)
        )

        # w/o residual support
        dm_nrs = DMaker(n_agents, n_states, adj, gamma=0.5)
        final_nrs, iters_nrs = _no_residual_run(dm_nrs, bpas, rels, weights, n_states)
        dist_nrs = _final_distribution(final_nrs, n_states)
        metrics["w/o residual support"]["acc"].append(accuracy(dist_nrs, true_state))
        metrics["w/o residual support"]["brier"].append(brier_score(dist_nrs, true_state))
        metrics["w/o residual support"]["ll"].append(log_likelihood(dist_nrs, true_state))
        metrics["w/o residual support"]["iters"].append(reasoning_path_length(iters_nrs))
        metrics["w/o residual support"]["trace"].append(0.0)

        # Algebraic averaging
        dist_aa = _algebraic_average(bpas, n_states)
        metrics["Algebraic averaging"]["acc"].append(accuracy(dist_aa, true_state))
        metrics["Algebraic averaging"]["brier"].append(brier_score(dist_aa, true_state))
        metrics["Algebraic averaging"]["ll"].append(log_likelihood(dist_aa, true_state))
        metrics["Algebraic averaging"]["iters"].append(0.0)
        metrics["Algebraic averaging"]["trace"].append(0.0)

    return methods, metrics


def _wilcoxon_p(full_arr, variant_arr):
    """Wilcoxon signed-rank test p-value, robust to ties / zero differences."""
    diffs = np.asarray(full_arr) - np.asarray(variant_arr)
    if np.allclose(diffs, 0):
        return 1.0
    try:
        _, p = wilcoxon(full_arr, variant_arr, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0


def run():
    print("=== Ablation Study ===")
    print(f"  Aggregating over {N_SEEDS} random seeds ...")

    # Aggregate metrics over seeds
    aggregate = None
    methods = None
    for seed in range(N_SEEDS):
        methods, metrics = _evaluate_seed(seed)
        if aggregate is None:
            aggregate = {m: {k: [] for k in metrics[m]} for m in methods}
        for m in methods:
            for k in metrics[m]:
                # mean across the 50 trials in this seed
                aggregate[m][k].append(float(np.mean(metrics[m][k])))

    # Compute mean ± std across seeds
    summary = {
        m: {k: (float(np.mean(v)), float(np.std(v))) for k, v in aggregate[m].items()}
        for m in methods
    }

    # Wilcoxon tests against the full D-MAKER (using per-seed accuracy/brier)
    p_values = {}
    for m in methods:
        if m == "Full D-MAKER":
            continue
        p_values[m] = {
            "acc": _wilcoxon_p(aggregate["Full D-MAKER"]["acc"], aggregate[m]["acc"]),
            "brier": _wilcoxon_p(aggregate["Full D-MAKER"]["brier"], aggregate[m]["brier"]),
        }

    out_path = tables_dir() / "table_ablation.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Method & Accuracy & Brier Score & Log-Likelihood & "
            "Iterations & Conflict Traceability\\\\\n"
        )
        for m in methods:
            acc = summary[m]["acc"]
            brier = summary[m]["brier"]
            ll = summary[m]["ll"]
            iters = summary[m]["iters"]
            trace = summary[m]["trace"]
            f.write(
                f"{m} & ${acc[0]:.3f}\\pm{acc[1]:.3f}$ "
                f"& ${brier[0]:.3f}\\pm{brier[1]:.3f}$ "
                f"& ${ll[0]:.3f}\\pm{ll[1]:.3f}$ "
                f"& ${iters[0]:.1f}\\pm{iters[1]:.1f}$ "
                f"& ${trace[0]*100:.1f}\\%$\\\\\n"
            )
        f.write("\nWilcoxon signed-rank test against Full D-MAKER (two-sided)\\\\\n")
        for m, p in p_values.items():
            f.write(
                f"{m}: p_acc={p['acc']:.4f}, p_brier={p['brier']:.4f}\\\\\n"
            )

    print(f"Saved {out_path}")
    for m in methods:
        acc = summary[m]["acc"]
        brier = summary[m]["brier"]
        trace = summary[m]["trace"]
        line = (
            f"  {m:32s} acc={acc[0]:.3f}±{acc[1]:.3f}  "
            f"brier={brier[0]:.3f}±{brier[1]:.3f}  trace={trace[0]*100:.1f}%"
        )
        if m in p_values:
            line += f"  p_brier={p_values[m]['brier']:.4f}"
        print(line)


if __name__ == "__main__":
    run()
