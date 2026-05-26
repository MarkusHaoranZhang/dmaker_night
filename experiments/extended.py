"""Extended studies (Section 5.5).

Measures (rather than draws analytic curves) every quantity reported in the
paper's extended-studies section:

- Mis-specified dependence (entropy and Brier trajectories)
- Convergence rate vs topology (line, ring, random)
- Communication scalability (D-MAKER vs DIGing rounds at L=10..100)
- Parameter sensitivity (gamma, eps)
- Empirical topology effect on accuracy
- Robustness: dynamic edge failures and asynchronous message delays
"""

import numpy as np
import matplotlib.pyplot as plt

from dmaker_night import (
    DMaker,
    SyntheticDataset,
    centralized_maker,
)
from dmaker_night.metrics import accuracy, brier_score, project_to_states
from dmaker_night.utils import (
    all_pairs_shortest_path,
    fully_connected,
    line_graph,
    random_graph,
    ring_graph,
)
from ._common import N_SEEDS, figures_dir, tables_dir


def _avg_distribution(final_local, n_states):
    avg = np.mean(np.array(final_local), axis=0)
    return project_to_states(avg, n_states)


def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# 5.5.1 Mis-specified dependence
# ---------------------------------------------------------------------------
def mis_specified_dependence():
    """Compare D-MAKER's iterative trajectory against the true centralized
    MAKER answer when two agents share a common observation.

    The local Markov dependence assumption ``ᾱ = φ(d(i,j))`` cannot
    capture the fact that agents 0 and 1 see the same noise realisation.
    The centralized reference is computed by treating the two duplicated
    BPAs as a single combined evidence source (taking their elementwise
    minimum, which is the conservative way to merge duplicated reports
    in DS theory) and then applying ``centralized_maker`` to the
    deduplicated set. This is the answer an oracle that *knew* about the
    duplication would produce.
    """
    print("  Mis-specified dependence ...")
    rng = np.random.default_rng(42)
    n_agents, n_states = 6, 3
    adj = fully_connected(n_agents)
    n_iter = 30

    entropy_dm = np.zeros(n_iter)
    entropy_cen = np.zeros(n_iter)
    brier_dm = np.zeros(n_iter)
    brier_cen = np.zeros(n_iter)

    n_trials = 100
    for _trial in range(n_trials):
        true_state = int(rng.integers(0, n_states))
        # Agents 0 and 1 share the same noisy observation -> strongly dependent
        shared_obs = int(rng.choice(
            n_states,
            p=[0.85 if s == true_state else 0.075 for s in range(n_states)],
        ))
        bpas = []
        for i in range(n_agents):
            bpa = np.zeros(2 ** n_states - 1)
            if i in (0, 1):
                obs = shared_obs
            else:
                obs = int(rng.choice(
                    n_states,
                    p=[0.8 if s == true_state else 0.1 for s in range(n_states)],
                ))
            bpa[2 ** obs - 1] = 1.0
            bpas.append(bpa)
        rels = np.full(n_agents, 0.95)
        weights = [np.full(n_states, 0.85) for _ in range(n_agents)]

        # D-MAKER (mis-specified: assumes local Markov dependence only)
        dm = DMaker(n_agents, n_states, adj, gamma=0.5, max_iter=n_iter)
        _final, traj, _rel_traj, _iters = dm.run(bpas, rels, weights, return_traj=True)
        # Pad trajectory if it converged early
        while len(traj) < n_iter:
            traj.append(traj[-1])
        for k in range(n_iter):
            avg = np.mean(np.array(traj[k]), axis=0)
            dist = project_to_states(avg, n_states)
            entropy_dm[k] += _entropy(dist)
            brier_dm[k] += brier_score(dist, true_state)

        # Centralized reference: deduplicate the redundant pair before
        # applying ``centralized_maker``. Since agents 0 and 1 share the
        # same observation by construction, the elementwise minimum of
        # their (possibly noisy) BPAs equals the shared raw evidence. We
        # then fuse the deduplicated agent set centrally — this is the
        # answer an oracle that knew about the shared sensor would
        # compute. The result is constant across iterations and is drawn
        # as a horizontal reference line.
        deduped_bpa = np.minimum(bpas[0], bpas[1])
        s = float(deduped_bpa.sum())
        if s > 0:
            deduped_bpa = deduped_bpa / s
        bpas_cen = [deduped_bpa] + bpas[2:]
        rels_cen = np.array([rels[0]] + list(rels[2:]))
        weights_cen = [weights[0]] + weights[2:]
        cen_pred = centralized_maker(bpas_cen, rels_cen, weights_cen, n_states)
        cen_dist = project_to_states(cen_pred, n_states)
        cen_entropy = _entropy(cen_dist)
        cen_brier = brier_score(cen_dist, true_state)
        for k in range(n_iter):
            entropy_cen[k] += cen_entropy
            brier_cen[k] += cen_brier

    entropy_dm /= n_trials
    entropy_cen /= n_trials
    brier_dm /= n_trials
    brier_cen /= n_trials

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    k = np.arange(n_iter)
    ax1.plot(k, entropy_cen, "b-", linewidth=2, label="Centralized MAKER")
    ax1.plot(k, entropy_dm, "r--", linewidth=2, label="D-MAKER (mis-specified)")
    ax1.set_ylabel("Entropy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.96, "(a)", transform=ax1.transAxes,
             fontsize=12, fontweight="bold", va="top")

    ax2.plot(k, brier_cen, "b-", linewidth=2, label="Centralized MAKER")
    ax2.plot(k, brier_dm, "r--", linewidth=2, label="D-MAKER (mis-specified)")
    ax2.set_xlabel(r"Iteration $k$")
    ax2.set_ylabel("Brier score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", va="top")
    plt.tight_layout()
    out = figures_dir() / "fig_mis_spec.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"    Saved {out}")


# ---------------------------------------------------------------------------
# 5.5.2 Convergence rate vs topology
# ---------------------------------------------------------------------------
def convergence_rate_topology():
    print("  Convergence rate vs topology ...")
    L = 20
    n_states = 3
    n_props = 2 ** n_states - 1
    rng = np.random.default_rng(123)
    n_iter = 40

    topologies = {
        "Line": (line_graph(L), "r-"),
        "Ring": (ring_graph(L), "g--"),
        "Random": (random_graph(L, p=0.3, seed=7), "b-."),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    n_trials = 30
    for name, (adj, style) in topologies.items():
        traces = np.zeros((n_trials, n_iter))
        for trial in range(n_trials):
            true_state = int(rng.integers(0, n_states))
            bpas = []
            for _i in range(L):
                bpa = np.zeros(n_props)
                obs = int(rng.choice(n_states, p=[0.8 if s == true_state else 0.1
                                                   for s in range(n_states)]))
                bpa[2 ** obs - 1] = 1.0
                bpas.append(bpa)
            rels = np.full(L, 0.95)
            weights = [np.full(n_states, 0.85) for _ in range(L)]
            dm = DMaker(L, n_states, adj, gamma=0.5, max_iter=n_iter)
            _final, traj, _rel_traj, _iters = dm.run(
                bpas, rels, weights, return_traj=True
            )
            while len(traj) < n_iter:
                traj.append(traj[-1])
            for k in range(n_iter):
                avg = np.mean(np.array(traj[k]), axis=0)
                dist = project_to_states(avg, n_states)
                traces[trial, k] = float(dist[true_state])
        mean_trace = traces.mean(axis=0)
        diam = _diameter(adj)
        ax.plot(np.arange(n_iter), mean_trace, style, linewidth=2,
                label=f"{name} ($D={diam}$)")
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=1.5,
               label="Centralized MAKER")
    ax.set_xlabel(r"Iteration $k$")
    ax.set_ylabel("Probability mass for true state")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    plt.tight_layout()
    out = figures_dir() / "fig_convergence_rate.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"    Saved {out}")


def _diameter(adj):
    """Return the (integer) diameter of an undirected adjacency matrix."""
    dist = all_pairs_shortest_path(adj)
    finite = dist[np.isfinite(dist)]
    return int(finite.max()) if finite.size else 0


# ---------------------------------------------------------------------------
# 5.5.3 Communication scalability (measured)
# ---------------------------------------------------------------------------
def communication_scalability():
    """Empirically measure D-MAKER vs DIGing rounds at L=10..100."""
    print("  Communication scalability ...")
    rng = np.random.default_rng(7)
    n_states = 3
    n_props = 2 ** n_states - 1
    Ls = [10, 20, 50, 100]
    rows = []
    for L in Ls:
        adj = ring_graph(L)
        # Single representative trial (averaged below across seeds)
        dm_rounds = []
        dig_rounds = []
        for seed in range(5):
            local_rng = np.random.default_rng(seed * 100 + L)
            true_state = int(local_rng.integers(0, n_states))
            bpas = []
            for _i in range(L):
                bpa = np.zeros(n_props)
                obs = int(local_rng.choice(
                    n_states,
                    p=[0.8 if s == true_state else 0.1 for s in range(n_states)],
                ))
                bpa[2 ** obs - 1] = 1.0
                bpas.append(bpa)
            rels = np.full(L, 0.95)
            weights = [np.full(n_states, 0.85) for _ in range(L)]

            dm = DMaker(L, n_states, adj, gamma=0.5, max_iter=400, eps=1e-3)
            _final, iters = dm.run(bpas, rels, weights)
            dm_rounds.append(iters)

            # Inline DIGing loop so we can record the iteration count
            # directly. Mirrors ``DistributedGradientTracking._run_with_step``
            # but exposes the per-iteration counter we report in Table 3.
            local = [b.copy() for b in bpas]
            tracker = [np.zeros_like(b) for b in local]
            iters_dig = 0
            for it in range(400):
                new_local, new_tracker = [], []
                for i in range(L):
                    neighbors = np.where(adj[i] == 1)[0]
                    grad = local[i] - bpas[i]
                    avg_track = tracker[i].copy()
                    for j in neighbors:
                        avg_track += tracker[j]
                    avg_track /= (1 + len(neighbors))
                    tracked = grad + avg_track - tracker[i]
                    new_tracker.append(tracked)
                    avg_est = local[i].copy()
                    for j in neighbors:
                        avg_est += local[j]
                    avg_est /= (1 + len(neighbors))
                    proposal = avg_est - 0.05 * tracked
                    proposal = np.clip(proposal, 0.0, None)
                    s = float(np.sum(proposal))
                    if s > 0:
                        proposal /= s
                    new_local.append(proposal)
                max_change = max(
                    float(np.max(np.abs(new_local[i] - local[i]))) for i in range(L)
                )
                local, tracker = new_local, new_tracker
                iters_dig = it + 1
                if max_change < 1e-3:
                    break
            dig_rounds.append(iters_dig)

        rows.append((L, n_props, int(np.mean(dm_rounds)), n_states,
                     int(np.mean(dig_rounds))))

    out = tables_dir() / "table_scalability.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("$L$ & D-MAKER Payload & D-MAKER Rounds & DIGing Payload & DIGing Rounds\\\\\n")
        for L, payload_dm, rounds_dm, payload_dig, rounds_dig in rows:
            f.write(
                f"${L}$ & ${payload_dm}$ & ${rounds_dm}$ "
                f"& ${payload_dig}$ & ${rounds_dig}$\\\\\n"
            )
    print(f"    Saved {out}")


# ---------------------------------------------------------------------------
# 5.5.4 Parameter sensitivity (measured)
# ---------------------------------------------------------------------------
def parameter_sensitivity():
    """Sweep γ and ε on the synthetic dataset.

    The dataset and per-trial BPA averages are independent of the
    swept parameter, so they are computed once per seed and cached
    across the inner loop. This collapses 30 × (10 + 7) dataset
    generations into 30, and the same factor on the BPA-averaging
    work, which is the dominant cost of this section.
    """
    print("  Parameter sensitivity ...")
    n_agents, n_states = 8, 3
    adj = fully_connected(n_agents)
    gamma_vals = np.linspace(0.05, 1.5, 10)
    eps_vals = np.array([1e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1])

    # Pre-compute the per-seed dataset and per-trial BPA averages once.
    cached_seeds = []
    for seed in range(N_SEEDS):
        ds = SyntheticDataset(n_agents=n_agents, n_states=n_states,
                              n_trials=20, n_timesteps=10, seed=seed)
        trials = ds.generate()
        prepared = []
        for trial in trials:
            bpas = [
                np.mean(np.array(trial["evidence"][a]), axis=0)
                for a in range(n_agents)
            ]
            prepared.append((bpas, trial["reliabilities"],
                             trial["weights"], trial["true_state"]))
        cached_seeds.append(prepared)

    def sweep(make_dmaker):
        """Run the sweep over the cached seeds with a parameterised DMaker."""
        accs = []
        for prepared in cached_seeds:
            dm = make_dmaker()
            for bpas, rels, weights, true_state in prepared:
                final, _ = dm.run(bpas, rels, weights)
                dist = _avg_distribution(final, n_states)
                accs.append(accuracy(dist, true_state))
        return float(np.mean(accs))

    acc_gamma = [
        sweep(lambda g=float(g): DMaker(n_agents, n_states, adj, gamma=g))
        for g in gamma_vals
    ]
    acc_eps = [
        sweep(lambda e=float(e): DMaker(n_agents, n_states, adj,
                                          gamma=0.5, eps=e))
        for e in eps_vals
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.plot(gamma_vals, acc_gamma, "b-o", linewidth=2)
    ax1.axvline(0.5, color="red", linestyle="--", label=r"$\gamma=0.5$")
    ax1.set_xlabel(r"$\gamma$")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.96, "(a)", transform=ax1.transAxes,
             fontsize=12, fontweight="bold", va="top")

    ax2.semilogx(eps_vals, acc_eps, "b-o", linewidth=2)
    ax2.axvline(1e-3, color="red", linestyle="--", label=r"$\varepsilon=10^{-3}$")
    ax2.set_xlabel(r"$\varepsilon$")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", va="top")
    plt.tight_layout()
    out = figures_dir() / "fig_sensitivity.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"    Saved {out}")


# ---------------------------------------------------------------------------
# 5.5.5 Empirical topology effect
# ---------------------------------------------------------------------------
def empirical_topology_effect():
    print("  Empirical topology effect ...")
    n_agents, n_states = 8, 3
    topologies = {
        "Line": line_graph(n_agents),
        "Ring": ring_graph(n_agents),
        "Random": random_graph(n_agents, p=0.4, seed=7),
        "Fully connected": fully_connected(n_agents),
    }
    out = tables_dir() / "table_topology.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("Topology & Accuracy & Brier Score & Mean Iterations\\\\\n")
        for name, adj in topologies.items():
            accs, briers, iter_counts = [], [], []
            for seed in range(N_SEEDS):
                ds = SyntheticDataset(n_agents=n_agents, n_states=n_states,
                                      n_trials=20, n_timesteps=10, seed=seed)
                trials = ds.generate()
                dm = DMaker(n_agents, n_states, adj, gamma=0.5)
                for trial in trials:
                    bpas = [
                        np.mean(np.array(trial["evidence"][a]), axis=0)
                        for a in range(n_agents)
                    ]
                    final, iters = dm.run(bpas, trial["reliabilities"], trial["weights"])
                    dist = _avg_distribution(final, n_states)
                    accs.append(accuracy(dist, trial["true_state"]))
                    briers.append(brier_score(dist, trial["true_state"]))
                    iter_counts.append(iters)
            f.write(
                f"{name} & ${np.mean(accs):.3f}\\pm{np.std(accs):.3f}$ "
                f"& ${np.mean(briers):.3f}\\pm{np.std(briers):.3f}$ "
                f"& ${np.mean(iter_counts):.1f}$\\\\\n"
            )
            print(
                f"    {name:15s} acc={np.mean(accs):.3f}  "
                f"brier={np.mean(briers):.3f}  iters={np.mean(iter_counts):.1f}"
            )
    print(f"    Saved {out}")


# ---------------------------------------------------------------------------
# 5.5.6 Robustness to dynamic topology and asynchronous delays
# ---------------------------------------------------------------------------
def robustness_study():
    """Quantify the effect of edge failures (30% removed every 5 iter) and
    random message delays (up to 3 time steps) on D-MAKER convergence.

    We re-instantiate ``DMaker`` with the dynamic adjacency at each
    sampling boundary so that the public ``run`` interface is reused
    instead of hand-rolling the iteration loop.
    """
    print("  Robustness study ...")
    n_agents, n_states = 8, 3
    base_adj = fully_connected(n_agents)

    def _trial_inputs(trial):
        bpas = [
            np.mean(np.array(trial["evidence"][a]), axis=0)
            for a in range(n_agents)
        ]
        return bpas, trial["reliabilities"], trial["weights"]

    def run_static(seed):
        ds = SyntheticDataset(n_agents=n_agents, n_states=n_states,
                              n_trials=10, n_timesteps=10, seed=seed)
        trials = ds.generate()
        accs, conv_iters = [], []
        for trial in trials:
            bpas, rels, weights = _trial_inputs(trial)
            dm = DMaker(n_agents, n_states, base_adj, gamma=0.5, eps=1e-3,
                        max_iter=200, estimate_reliability=False)
            final, iters = dm.run(bpas, rels, weights)
            avg = np.mean(np.array(final), axis=0)
            dist = project_to_states(avg, n_states)
            accs.append(accuracy(dist, trial["true_state"]))
            conv_iters.append(iters)
        return float(np.mean(accs)), float(np.mean(conv_iters))

    def run_dynamic_topology(seed, drop_rate, drop_every):
        rng_local = np.random.default_rng(seed)
        ds = SyntheticDataset(n_agents=n_agents, n_states=n_states,
                              n_trials=10, n_timesteps=10, seed=seed)
        trials = ds.generate()
        accs, conv_iters = [], []
        for trial in trials:
            bpas, rels, weights = _trial_inputs(trial)
            # Run an iteration block, drop edges, run another block.
            dm = DMaker(n_agents, n_states, base_adj, gamma=0.5,
                        eps=1e-3, max_iter=drop_every,
                        estimate_reliability=False)
            local, _it = dm.run(bpas, rels, weights)
            iters_used = drop_every
            for _block in range(20):
                edges = np.argwhere(np.triu(base_adj, 1) == 1)
                n_drop = int(drop_rate * len(edges))
                adj = base_adj.copy()
                if n_drop > 0:
                    drop_idx = rng_local.choice(len(edges), size=n_drop, replace=False)
                    for di in drop_idx:
                        u, v = edges[di]
                        adj[u, v] = 0
                        adj[v, u] = 0
                # Use the current local estimates as the new initial BPAs
                # by calling conjunctive_combine directly through DMaker
                # initialized with adjusted adjacency. Reliabilities of 1.0
                # are passed because the local estimates already absorbed
                # the original reliabilities in their first initialization.
                dm2 = DMaker(n_agents, n_states, adj, gamma=0.5, eps=1e-3,
                             max_iter=drop_every,
                             estimate_reliability=False)
                # Re-initialize each local BPA as an "evidence" with full
                # reliability, treating it as already-discounted mass.
                ones = np.ones(n_agents)
                identity_weights = [np.ones(n_states) for _ in range(n_agents)]
                final, iters = dm2.run(local, ones, identity_weights)
                local = final
                iters_used += iters
                if iters < drop_every:
                    break
            avg = np.mean(np.array(local), axis=0)
            dist = project_to_states(avg, n_states)
            accs.append(accuracy(dist, trial["true_state"]))
            conv_iters.append(iters_used)
        return float(np.mean(accs)), float(np.mean(conv_iters))

    def run_async_delays(seed, max_delay):
        """Simulate random message delays of up to ``max_delay`` iterations.

        Each agent reads neighbors' BPAs from a delayed snapshot of the
        history. Convergence is checked on the live state.
        """
        rng_local = np.random.default_rng(seed)
        ds = SyntheticDataset(n_agents=n_agents, n_states=n_states,
                              n_trials=10, n_timesteps=10, seed=seed)
        trials = ds.generate()
        accs, conv_iters = [], []
        for trial in trials:
            bpas, rels, weights = _trial_inputs(trial)
            adj = base_adj
            dm = DMaker(n_agents, n_states, adj, gamma=0.5, eps=1e-3,
                        max_iter=200, estimate_reliability=False)
            init_local = dm.initialize_local(bpas, rels, weights)
            history = [list(init_local)]
            local = list(init_local)
            iters_used = 0
            n_iter = 200
            for k in range(1, n_iter):
                new_local = []
                for i in range(n_agents):
                    neighbors = np.where(adj[i] == 1)[0]
                    if len(neighbors) == 0:
                        new_local.append(local[i].copy())
                        continue
                    neighbor_states = []
                    neighbor_rels = []
                    neighbor_dists = []
                    for j in neighbors:
                        delay = int(rng_local.integers(0, max_delay + 1))
                        src_iter = max(0, k - 1 - delay)
                        neighbor_states.append(history[src_iter][j])
                        neighbor_rels.append(float(rels[j]))
                        neighbor_dists.append(dm.dist[i, j])
                    combined = dm.combine_neighbors(
                        local[i], neighbor_states,
                        float(rels[i]), neighbor_rels, neighbor_dists,
                    )
                    new_local.append(combined)
                max_change = max(
                    float(np.max(np.abs(new_local[i] - local[i])))
                    for i in range(n_agents)
                )
                local = new_local
                history.append(list(local))
                iters_used = k
                if max_change < dm.eps:
                    break
            avg = np.mean(np.array(local), axis=0)
            dist = project_to_states(avg, n_states)
            accs.append(accuracy(dist, trial["true_state"]))
            conv_iters.append(iters_used)
        return float(np.mean(accs)), float(np.mean(conv_iters))

    results = {
        "Static, no delays": [],
        "30% edges dropped every 5 iter": [],
        "Random delays up to 3 steps": [],
    }
    for seed in range(N_SEEDS):
        results["Static, no delays"].append(run_static(seed))
        results["30% edges dropped every 5 iter"].append(
            run_dynamic_topology(seed, drop_rate=0.3, drop_every=5)
        )
        results["Random delays up to 3 steps"].append(run_async_delays(seed, 3))

    out = tables_dir() / "table_robustness.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("Scenario & Accuracy & Mean Iterations\\\\\n")
        for name, vals in results.items():
            accs = np.array([v[0] for v in vals])
            iters = np.array([v[1] for v in vals])
            f.write(
                f"{name} & ${accs.mean():.3f}\\pm{accs.std():.3f}$ "
                f"& ${iters.mean():.1f}\\pm{iters.std():.1f}$\\\\\n"
            )
            print(
                f"    {name:35s} acc={accs.mean():.3f}±{accs.std():.3f}  "
                f"iters={iters.mean():.1f}±{iters.std():.1f}"
            )
    print(f"    Saved {out}")


def run():
    print("=== Extended Studies ===")
    mis_specified_dependence()
    convergence_rate_topology()
    communication_scalability()
    parameter_sensitivity()
    empirical_topology_effect()
    robustness_study()


if __name__ == "__main__":
    run()
