"""Preliminary study (Section 5.2): convergence and reliability suppression on a 3-agent line graph."""

import numpy as np
import matplotlib.pyplot as plt

from dmaker_night import DMaker, centralized_maker
from dmaker_night.metrics import project_to_states
from ._common import figures_dir


def run():
    print("=== Preliminary Study ===")
    n_agents, n_states = 3, 3
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    rels = np.array([0.95, 0.95, 0.95])
    weights = [
        np.array([0.90, 0.05, 0.05]),
        np.array([0.88, 0.07, 0.05]),
        np.array([0.05, 0.90, 0.05]),  # conflicting agent
    ]
    bpas = [np.zeros(2 ** n_states - 1) for _ in range(n_agents)]
    bpas[0][0] = 1.0  # supports A (singleton index = 1 - 1)
    bpas[1][0] = 1.0  # supports A
    bpas[2][1] = 1.0  # supports B (conflict)

    dm = DMaker(n_agents, n_states, adj, max_iter=25)
    final, traj, rel_traj_list, iters = dm.run(
        bpas, rels, weights, return_traj=True
    )
    print(f"Converged in {iters} iterations.")

    # Reference line: probability mass that centralized MAKER assigns to the
    # true state (A) on the same input. Computed dynamically so the figure
    # stays correct if anyone retunes ``rels`` or ``weights`` above.
    cen_pred = centralized_maker(bpas, rels, weights, n_states)
    cen_dist = project_to_states(cen_pred, n_states)
    cen_true_state_mass = float(cen_dist[0])  # state A is index 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    colors = ["#2196F3", "#4CAF50", "#F44336"]
    labels = ["Reliable Agent 1", "Reliable Agent 2", "Conflicting Agent"]

    # Panel (a): probability mass on the true state (A) per iteration
    for i in range(n_agents):
        probs = [step[i][0] for step in traj]
        ax1.plot(range(len(traj)), probs, color=colors[i], label=labels[i], linewidth=2)
    ax1.axhline(cen_true_state_mass, color="gray", linestyle=":",
                label="Centralized MAKER", linewidth=1.5)
    ax1.set_ylabel("Probability mass for true state")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.96, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="top")

    # Panel (b): estimated reliability trajectories from the protocol itself
    rel_traj = np.array(rel_traj_list)  # shape (n_steps, n_agents)
    n_steps = rel_traj.shape[0]
    for i in range(n_agents):
        ax2.plot(range(n_steps), rel_traj[:, i], color=colors[i], label=labels[i], linewidth=2)
    ax2.set_xlabel(r"Iteration $k$")
    ax2.set_ylabel(r"Estimated reliability $r_i$")
    ax2.legend(loc="lower right")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top")

    plt.tight_layout()
    out = figures_dir() / "fig_convergence.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    run()
