"""Comparative study (Section 5.4): D-MAKER vs baselines across three datasets.

Each method is evaluated on 30 independent random seeds. Decision-quality
metrics (accuracy, F1, Brier, log-likelihood) and interpretability metrics
(weight-reliability separation, conflict traceability, reasoning path length)
are averaged across seeds. Pairwise Wilcoxon signed-rank tests against
Centralized MAKER are reported.

Also produces ``fig_scatter.pdf`` (Figure 2 in the paper).

When the optional ``DMAKER_NIGHT_USE_REAL_UCI=1`` environment variable is set,
the UCI Gas Sensor Array Drift dataset is downloaded from the UCI ML
repository and used in place of the procedural simulator.
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from dmaker_night import (
    DMaker,
    CentralizedMAKER,
    DempsterConsensus,
    DistributedGradientTracking,
    DistributedBayesianFilter,
    MajorityVoting,
    SyntheticDataset,
    UCIGasDataset,
    AirSimUAVDataset,
)
from dmaker_night.metrics import (
    accuracy,
    brier_score,
    f1_score,
    log_likelihood,
    project_to_states,
    reasoning_path_length,
    weight_reliability_separation,
    conflict_traceability,
)
from dmaker_night.utils import fully_connected, ring_graph
from ._common import N_SEEDS, figures_dir, tables_dir


METHOD_ORDER = [
    "Centralized MAKER",
    "D-MAKER",
    "Dempster+Consensus",
    "Gradient tracking",
    "Bayesian filtering",
    "Majority voting",
]


_USE_REAL_UCI = os.environ.get("DMAKER_NIGHT_USE_REAL_UCI", "").lower() in ("1", "true", "yes")


def _load_uci_real_or_simulated(seed):
    """Return a :class:`UCIGasDataset`-shaped trial list.

    Tries the real UCI dataset when ``DMAKER_NIGHT_USE_REAL_UCI=1``; falls back
    to the procedural simulator otherwise (or on download failure).
    """
    if _USE_REAL_UCI:
        try:
            from dmaker_night.datasets_real import load_uci_gas_evidence
            trials, _, _ = load_uci_gas_evidence(
                n_trials=30, n_sensors=16, n_states=6, seed=seed,
            )
            return trials, 16, 6
        except Exception as exc:  # pragma: no cover - network dependent
            warnings.warn(
                f"Real UCI Gas dataset unavailable, falling back to simulator: {exc}"
            )
    ds = UCIGasDataset(n_agents=16, n_states=6, n_trials=30, n_timesteps=5, seed=seed)
    return ds.generate(), 16, 6


def _trial_inputs(trial, n_agents):
    bpas = [np.mean(np.array(trial["evidence"][a]), axis=0) for a in range(n_agents)]
    rel = trial["reliabilities"]
    rel_scalar = rel.mean(axis=0) if rel.ndim == 2 else rel
    return bpas, rel_scalar


def _expand_trials_per_timestep(trials, n_agents):
    """Expand a trial list with ``T`` time steps into ``len(trials) * T``
    independent single-step trials.

    Section 5.1.1 of the paper specifies "50 independent trials, each with
    10 time steps". We treat each (trial, time-step) pair as a single
    independent observation in the comparative study, which yields the
    aggregate accuracies reported in Table 2.
    """
    expanded = []
    for trial in trials:
        evidence = trial["evidence"]
        rel = np.asarray(trial["reliabilities"])
        weights = trial["weights"]
        n_t = len(evidence[0])
        for t in range(n_t):
            bpas_t = [np.asarray(evidence[a][t], dtype=float) for a in range(n_agents)]
            if rel.ndim == 2:
                rel_t = rel[t]
            else:
                rel_t = rel
            expanded.append({
                "true_state": trial["true_state"],
                "evidence": [[bpas_t[a]] for a in range(n_agents)],
                "reliabilities": np.asarray(rel_t, dtype=float),
                "weights": weights,
                "agent_types": trial.get("agent_types"),
            })
    return expanded


def _evaluate_dataset(trial_loader, adjacency_fn, conflicting_idx, seed):
    """Run all methods on one dataset realization and return per-trial metrics.

    ``trial_loader(seed)`` is a callable that returns ``(trials, n_agents,
    n_states)`` so we can swap simulated and real datasets transparently.
    Each (trial, time-step) pair is treated as an independent observation,
    matching the paper's "50 independent trials, each with 10 time steps"
    specification (Section 5.1.1).
    """
    raw_trials, n_agents, n_states = trial_loader(seed)
    trials = _expand_trials_per_timestep(raw_trials, n_agents)
    adjacency = adjacency_fn(n_agents)

    methods = {
        "Centralized MAKER": CentralizedMAKER(n_states=n_states),
        "D-MAKER": DMaker(n_agents, n_states, adjacency, gamma=0.5),
        "Dempster+Consensus": DempsterConsensus(n_agents, adjacency),
        "Gradient tracking": DistributedGradientTracking(n_agents, adjacency),
        "Bayesian filtering": DistributedBayesianFilter(n_agents, adjacency),
        "Majority voting": MajorityVoting(),
    }
    results = {
        m: {"acc": [], "f1": [], "brier": [], "ll": [],
            "iters": [], "wr": [], "trace": []}
        for m in methods
    }

    for trial in trials:
        true_state = trial["true_state"]
        bpas, rel_scalar = _trial_inputs(trial, n_agents)
        weights = trial["weights"]
        # Ground-truth sensor quality is the agent's inherent weight on
        # the *true* state. This depends only on the dataset's confusion
        # matrices (or, for the real UCI loader, on the per-class
        # accuracy of the Gaussian likelihood on the training fold) —
        # never on the BPA that is also fed into the inference methods.
        # That independence is what makes the W-R separation metric a
        # meaningful test of how well a method's est_rel tracks an
        # observable, *external* notion of sensor quality.
        gt_quality = np.array(
            [float(weights[a][true_state]) for a in range(n_agents)]
        )

        for m_name, method in methods.items():
            iters = 0
            est_rel = None
            if m_name == "Centralized MAKER":
                pred = method.run(bpas, rel_scalar, weights)
                est_rel = rel_scalar
                iters = 1  # one-shot combination
            elif m_name == "D-MAKER":
                final_local, iters, est_rel = method.run(
                    bpas, rel_scalar, weights, return_estimated_reliability=True,
                )
                pred = np.mean(np.array(final_local), axis=0)
            elif m_name == "Gradient tracking":
                # Step size selected from the {0.01, 0.05, 0.1, 0.5}
                # grid via internal validation, per Section 5.1.4.
                pred = method.run_with_grid_search(bpas, rel_scalar)
            elif m_name == "Bayesian filtering":
                # Process noise selected from the {0.01, 0.1, 1.0} grid
                # via internal validation, per Section 5.1.4.
                pred = method.run_with_grid_search(bpas, rel_scalar)
            else:
                pred = method.run(bpas, rel_scalar)

            dist = project_to_states(pred, n_states)
            results[m_name]["acc"].append(accuracy(dist, true_state))
            results[m_name]["f1"].append(f1_score(dist, true_state))
            results[m_name]["brier"].append(brier_score(dist, true_state))
            results[m_name]["ll"].append(log_likelihood(dist, true_state))
            results[m_name]["iters"].append(reasoning_path_length(iters))
            if est_rel is not None and len(est_rel) == n_agents:
                wr = weight_reliability_separation(est_rel, gt_quality)
                results[m_name]["wr"].append(wr)
                if conflicting_idx is not None:
                    results[m_name]["trace"].append(
                        conflict_traceability(est_rel, conflicting_idx)
                    )
    return results


def _wilcoxon_p(a, b):
    diffs = np.asarray(a) - np.asarray(b)
    if np.allclose(diffs, 0):
        return 1.0
    try:
        _, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0


def _aggregate_dataset(trial_loader, adjacency_fn, conflicting_idx):
    """Aggregate metrics across N_SEEDS dataset realizations.

    Returns per-method (mean, std) for every metric and per-seed accuracy/brier
    arrays for Wilcoxon tests.
    """
    seed_metrics = {m: {} for m in METHOD_ORDER}
    for seed in range(N_SEEDS):
        results = _evaluate_dataset(
            trial_loader, adjacency_fn, conflicting_idx, seed
        )
        for m_name in METHOD_ORDER:
            for k, vals in results[m_name].items():
                seed_metrics[m_name].setdefault(k, [])
                if vals:
                    seed_metrics[m_name][k].append(float(np.mean(vals)))

    summary = {}
    for m in METHOD_ORDER:
        summary[m] = {}
        for k, vs in seed_metrics[m].items():
            if vs:
                summary[m][k] = (float(np.mean(vs)), float(np.std(vs)))
    return summary, seed_metrics


def _format_metric(summary, key, fmt="{:.3f}"):
    if key not in summary or not summary[key]:
        return "N/A"
    mean, std = summary[key]
    return fmt.format(mean)


def run():
    print("=== Comparative Study ===")
    print(f"  Aggregating over {N_SEEDS} random seeds per dataset ...")
    if _USE_REAL_UCI:
        print("  Using REAL UCI Gas Sensor Array Drift dataset (download on first use)")
    else:
        print("  Using simulated UCI Gas data (set DMAKER_NIGHT_USE_REAL_UCI=1 for the real dataset)")

    def synthetic_loader(seed):
        ds = SyntheticDataset(n_agents=8, n_states=3, n_trials=50, n_timesteps=10, seed=seed)
        return ds.generate(), ds.n_agents, ds.n_states

    def uav_loader(seed):
        # The AirSim + ResNet-18 BPA archive is deterministic and the
        # ``seed`` argument is accepted for interface compatibility with
        # the synthetic and UCI loaders. We pass it through so that
        # different seeds reshuffle the trial subset deterministically.
        ds = AirSimUAVDataset(n_trials=30, seed=seed)
        trials = ds.generate()
        rng = np.random.default_rng(seed)
        rng.shuffle(trials)
        return trials, ds.n_agents, ds.n_states

    sections = [
        ("Synthetic", synthetic_loader, fully_connected, 7),
        ("UCI Gas", _load_uci_real_or_simulated, ring_graph, None),
        ("UAV Swarm", uav_loader, fully_connected, None),
    ]

    out_path = tables_dir() / "table_comparative.txt"
    scatter_points = {}  # method -> list of (decision_quality, interpretability)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Method & Accuracy & F1 & Brier & Log-Lik & "
            "Iters & W-R Sep & Traceability\\\\\n"
        )
        for name, loader, adj_fn, conf_idx in sections:
            print(f"-- {name} --")
            summary, seed_metrics = _aggregate_dataset(loader, adj_fn, conf_idx)
            f.write(f"\\multicolumn{{8}}{{c}}{{{name}}}\\\\\n")
            for m in METHOD_ORDER:
                s = summary[m]
                acc = _format_metric(s, "acc")
                f1 = _format_metric(s, "f1")
                brier = _format_metric(s, "brier")
                ll = _format_metric(s, "ll")
                iters = _format_metric(s, "iters", "{:.1f}")
                wr = _format_metric(s, "wr", "{:.2f}")
                tr = _format_metric(s, "trace", "{:.2f}")
                f.write(f"{m} & {acc} & {f1} & {brier} & {ll} & {iters} & {wr} & {tr}\\\\\n")
                print(
                    f"  {m:22s} acc={acc}  f1={f1}  brier={brier}  "
                    f"ll={ll}  iters={iters}  wr={wr}  trace={tr}"
                )
                if "acc" in s:
                    interp_components = []
                    if "wr" in s:
                        interp_components.append(max(0.0, s["wr"][0]))
                    if "trace" in s:
                        interp_components.append(s["trace"][0])
                    interp = float(np.mean(interp_components)) if interp_components else 0.0
                    scatter_points.setdefault(m, []).append((s["acc"][0], interp, name))

            # Wilcoxon vs Centralized MAKER
            f.write(f"\\multicolumn{{8}}{{l}}{{Wilcoxon vs Centralized MAKER (acc / brier)}}\\\\\n")
            for m in METHOD_ORDER:
                if m == "Centralized MAKER":
                    continue
                if not seed_metrics["Centralized MAKER"].get("acc") or not seed_metrics[m].get("acc"):
                    continue
                p_acc = _wilcoxon_p(
                    seed_metrics["Centralized MAKER"]["acc"], seed_metrics[m]["acc"]
                )
                p_br = _wilcoxon_p(
                    seed_metrics["Centralized MAKER"]["brier"], seed_metrics[m]["brier"]
                )
                f.write(f"{m} & p_acc={p_acc:.4f}, p_brier={p_br:.4f}\\\\\n")
    print(f"Saved {out_path}")

    _plot_scatter(scatter_points)


def _plot_scatter(scatter_points):
    """Figure 2: decision quality vs interpretability."""
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {
        "Centralized MAKER": ("D", "#000000"),
        "D-MAKER": ("o", "#D81B60"),
        "Dempster+Consensus": ("s", "#1E88E5"),
        "Gradient tracking": ("^", "#FFC107"),
        "Bayesian filtering": ("v", "#43A047"),
        "Majority voting": ("X", "#8E24AA"),
    }
    for method, points in scatter_points.items():
        marker, color = markers.get(method, ("o", "gray"))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, marker=marker, color=color, s=120, label=method,
                   edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.set_xlabel("Decision quality (accuracy)")
    ax.set_ylabel("Interpretability (mean of W-R separation and traceability)")
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    out = figures_dir() / "fig_scatter.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    run()
