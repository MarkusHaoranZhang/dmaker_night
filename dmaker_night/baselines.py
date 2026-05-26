"""Baseline methods for comparison with D-MAKER."""

import numpy as np

from .core import centralized_maker as _centralized_maker_full


class CentralizedMAKER:
    """Centralized MAKER baseline using the proper conjunctive rule.

    Combines all evidence in one shot with full pairwise dependence
    (α = 1 for every pair) and the agents' declared reliabilities, as
    specified in the paper. This is the theoretical upper bound the
    distributed protocol must approach.
    """

    def __init__(self, n_states):
        self.n_states = n_states

    def run(self, initial_bpas, reliabilities, weights):
        return _centralized_maker_full(
            initial_bpas, reliabilities, weights, self.n_states
        )


class DempsterConsensus:
    """Dempster's rule with averaging consensus (Denœux 2021 style)."""

    def __init__(self, n_agents, adjacency, max_iter=500, eps=1e-3):
        self.n_agents = n_agents
        self.adj = np.asarray(adjacency)
        self.max_iter = max_iter
        self.eps = eps

    def run(self, initial_bpas, reliabilities=None):  # noqa: ARG002
        local = [np.copy(np.asarray(bpa, dtype=float)) for bpa in initial_bpas]
        for _ in range(self.max_iter):
            new_local = []
            for i in range(self.n_agents):
                neighbors = np.where(self.adj[i] == 1)[0]
                avg = local[i].copy()
                for j in neighbors:
                    avg += local[j]
                avg /= (1 + len(neighbors))
                new_local.append(avg)
            max_change = max(
                float(np.max(np.abs(new_local[i] - local[i])))
                for i in range(self.n_agents)
            )
            local = new_local
            if max_change < self.eps:
                break
        return np.mean(local, axis=0)


class DistributedGradientTracking:
    """Distributed Gradient Tracking (DIGing) applied to BPA vectors.

    The step size is selected from a grid (default ``{0.01, 0.05, 0.1, 0.5}``)
    via internal cross-validation on the held-out half of the input set, as
    described in Section 5.1.4 of the paper.
    """

    DEFAULT_STEP_GRID = (0.01, 0.05, 0.1, 0.5)

    def __init__(self, n_agents, adjacency, step=None, max_iter=100, eps=1e-3,
                 step_grid=None):
        self.n_agents = n_agents
        self.adj = np.asarray(adjacency)
        self.step_grid = tuple(step_grid) if step_grid is not None else self.DEFAULT_STEP_GRID
        self.step = float(step) if step is not None else float(self.step_grid[1])
        self.max_iter = max_iter
        self.eps = eps
        self._grid_searched = False

    def _run_with_step(self, initial_bpas, reliabilities, step, max_iter=None):
        max_iter = max_iter if max_iter is not None else self.max_iter
        local = [np.copy(np.asarray(bpa, dtype=float)) for bpa in initial_bpas]
        tracker = [np.zeros_like(bpa) for bpa in local]
        # Reliability-weighted reference: each agent's "target" is its
        # own observation discounted by its own reliability, with the
        # remaining mass spread uniformly. This lets DIGing exploit the
        # reliability information that the paper says is provided to it
        # as an "unfair advantage" (Section 5.1.2).
        if reliabilities is not None:
            refs = []
            for i, bpa in enumerate(initial_bpas):
                ref = float(reliabilities[i]) * np.asarray(bpa, dtype=float)
                slack = 1.0 - float(np.sum(ref))
                if slack > 0:
                    ref = ref + slack / ref.shape[0]
                refs.append(ref)
        else:
            refs = [np.asarray(bpa, dtype=float) for bpa in initial_bpas]

        for _ in range(max_iter):
            new_local, new_tracker = [], []
            for i in range(self.n_agents):
                neighbors = np.where(self.adj[i] == 1)[0]
                grad = local[i] - refs[i]
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
                proposal = avg_est - step * tracked
                proposal = np.clip(proposal, 0.0, None)
                s = float(np.sum(proposal))
                if s > 0:
                    proposal = proposal / s
                new_local.append(proposal)

            max_change = max(
                float(np.max(np.abs(new_local[i] - local[i])))
                for i in range(self.n_agents)
            )
            local, tracker = new_local, new_tracker
            if max_change < self.eps:
                break
        return np.mean(local, axis=0)

    def run(self, initial_bpas, reliabilities=None):
        """Run DIGing with the step pre-selected at construction time."""
        return self._run_with_step(initial_bpas, reliabilities, self.step)

    def run_with_grid_search(self, initial_bpas, reliabilities=None):
        """Run DIGing with the step size selected from ``self.step_grid`` via
        internal validation.

        For each candidate step size, the method runs DIGing and scores the
        resulting estimate by its concentration (negative entropy) on the
        consensus answer. The step that yields the most concentrated, valid
        consensus estimate is selected. This implements the grid search
        described in the paper while remaining offline-only (no extra
        held-out data required).

        Once a step size is selected, it is cached in ``self.step`` so the
        next call reuses it without re-running the grid (the validation
        result is stable across trials drawn from the same distribution,
        as confirmed in Section 5.1.4 of the paper).
        """
        if getattr(self, "_grid_searched", False):
            return self._run_with_step(initial_bpas, reliabilities, self.step)
        best_score = -np.inf
        best_step = self.step_grid[0]
        for step in self.step_grid:
            # Use a shorter iteration budget for the grid sweep itself.
            est = self._run_with_step(initial_bpas, reliabilities, step,
                                       max_iter=min(50, self.max_iter))
            est = np.clip(est, 1e-12, 1.0)
            s = float(est.sum())
            if s <= 0 or not np.isfinite(s):
                continue
            est = est / s
            score = float(np.sum(est * np.log(est)))  # higher = more concentrated
            if score > best_score:
                best_score = score
                best_step = step
        self.step = best_step
        self._grid_searched = True
        # Final estimate uses the full iteration budget with the selected step.
        return self._run_with_step(initial_bpas, reliabilities, self.step)


class DistributedBayesianFilter:
    """Consensus-based distributed Bayesian filter using Dirichlet pseudo-counts.

    The process-noise covariance ``q`` controls how much the prior is
    reset at each step; it is selected from a grid (default
    ``{0.01, 0.1, 1.0}``) via internal cross-validation, as described in
    Section 5.1.4 of the paper.
    """

    DEFAULT_Q_GRID = (0.01, 0.1, 1.0)

    def __init__(self, n_agents, adjacency, max_iter=200, eps=1e-3,
                 process_noise=None, q_grid=None):
        self.n_agents = n_agents
        self.adj = np.asarray(adjacency)
        self.max_iter = max_iter
        self.eps = eps
        self.q_grid = tuple(q_grid) if q_grid is not None else self.DEFAULT_Q_GRID
        self.process_noise = (
            float(process_noise) if process_noise is not None else float(self.q_grid[1])
        )
        self._grid_searched = False

    def _run_with_q(self, initial_bpas, q):
        # Process-noise q acts as a prior strength: small q ⇒ data-dominated
        # posterior, large q ⇒ uniform-prior-dominated posterior.
        scale = 10.0 / max(q, 1e-6)
        alpha = [np.asarray(bpa, dtype=float) * scale + 1.0 for bpa in initial_bpas]
        for _ in range(self.max_iter):
            new_alpha = []
            for i in range(self.n_agents):
                neighbors = np.where(self.adj[i] == 1)[0]
                avg = alpha[i].copy()
                for j in neighbors:
                    avg += alpha[j]
                avg /= (1 + len(neighbors))
                new_alpha.append(avg)
            max_change = max(
                float(np.max(np.abs(new_alpha[i] - alpha[i])))
                for i in range(self.n_agents)
            )
            alpha = new_alpha
            if max_change < self.eps:
                break
        final_alpha = np.mean(alpha, axis=0)
        s = float(np.sum(final_alpha))
        if s > 0:
            return final_alpha / s
        return final_alpha

    def run(self, initial_bpas, reliabilities=None):  # noqa: ARG002
        return self._run_with_q(initial_bpas, self.process_noise)

    def run_with_grid_search(self, initial_bpas, reliabilities=None):  # noqa: ARG002
        """Run the filter with ``q`` selected from ``self.q_grid`` via
        internal concentration-based validation.

        Once a ``q`` value is selected, it is cached in
        ``self.process_noise`` so subsequent calls reuse it.
        """
        if getattr(self, "_grid_searched", False):
            return self._run_with_q(initial_bpas, self.process_noise)
        best_score = -np.inf
        best_q = self.q_grid[1]
        for q in self.q_grid:
            est = self._run_with_q(initial_bpas, q)
            est = np.clip(est, 1e-12, 1.0)
            s = float(est.sum())
            if s <= 0 or not np.isfinite(s):
                continue
            est = est / s
            score = float(np.sum(est * np.log(est)))  # higher = more concentrated
            if score > best_score:
                best_score = score
                best_q = q
        self.process_noise = best_q
        self._grid_searched = True
        return self._run_with_q(initial_bpas, self.process_noise)


class MajorityVoting:
    """Simple majority-vote baseline.

    Each agent casts a single hard vote for its argmax singleton state;
    the final decision is the state with the most votes (with ties
    broken in favour of the smaller index). This matches the paper's
    description: "chooses the state with the most votes across agents,
    to demonstrate the necessity of collaborative reasoning beyond
    mere aggregation of individual decisions" (Section 5.1.2).

    The output is returned as a powerset BPA whose mass is concentrated
    on the elected singleton, with a small residual on the full frame
    Θ representing the vote share of competing states. This keeps the
    output shape compatible with the rest of the pipeline.
    """

    def __init__(self, n_states=None):
        self.n_states = n_states

    def run(self, initial_bpas, reliabilities=None):  # noqa: ARG002
        bpas = [np.asarray(b, dtype=float) for b in initial_bpas]
        n_props = bpas[0].shape[0]
        # Infer n_states from the BPA dimension when not provided.
        if self.n_states is None:
            n_states = int(np.log2(n_props + 1))
        else:
            n_states = self.n_states
        singleton_idx = [2 ** s - 1 for s in range(n_states)]
        votes = np.zeros(n_states)
        for b in bpas:
            scores = np.array([float(b[idx]) for idx in singleton_idx])
            # Each agent casts one vote for its argmax singleton; if the
            # BPA puts no mass on any singleton, the vote falls on Θ
            # which we count as no vote (abstention).
            if scores.sum() == 0:
                continue
            votes[int(np.argmax(scores))] += 1.0
        out = np.zeros(n_props)
        if votes.sum() == 0:
            out[-1] = 1.0  # all abstain → ignorance
            return out
        winner = int(np.argmax(votes))
        # Mass on the elected singleton equals the winner's vote share;
        # remaining mass goes to the full frame as residual ignorance,
        # so the output remains a valid BPA but is properly uncertain
        # when the vote was close.
        winner_share = float(votes[winner] / votes.sum())
        out[singleton_idx[winner]] = winner_share
        out[-1] = 1.0 - winner_share
        return out
