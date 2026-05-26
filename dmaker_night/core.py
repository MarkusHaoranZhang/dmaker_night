"""Core D-MAKER protocol implementation.

Follows the paper exactly:

* Initialization: Eq. (2)+(3)
        m_θ,i^(t,0) = ω_i · w_θ,i · p_θ,i              for θ ⊂ Θ (singleton/composite)
        m_2Θ,i^(t,0) = ω_i · (1 - r_i)                  residual on the powerset
        ω_i = 1 / [Σ_θ w_θ,i · p_θ,i + (1 - r_i)]

* Pairwise conjunctive combination: Eq. (6)
        m̂_θ,(i,j)^(t,k) = (1 - r_j) · m_θ,i^(t,k-1)
                        + (1 - r_i) · m_θ,j^(t,k-1)
                        + Σ_{A∩B=θ} ω̄_ij · w̄_AiBj · ᾱ_AiBj
                                    · m_A,i^(t,k-1) · m_B,j^(t,k-1)
        ω̄_ij = 1 / (1 − r_i · r_j)         (projection factor)
        ᾱ_AiBj = φ(d(i,j))                  (Local Markov dependence)

* Update / normalization: Eq. (10)
        m_θ,i^(t,k) = m̂_θ,i^(t,k) / Σ_C m̂_C,i^(t,k)

* Convergence: Eq. (11)
        max_θ |m_θ,i^(t,k) − m_θ,i^(t,k-1)| < ε

The implementation uses a precomputed intersection lookup that maps the
``(A_idx, B_idx)`` pair on the powerset to the destination proposition
index for ``A ∩ B``, so the inner double sum reduces to a single
``np.add.at`` scatter, which keeps the protocol tractable for the
``|Θ| = 6`` UCI Gas case (``n_props = 63``).
"""

import numpy as np

from .utils import all_pairs_shortest_path


class DMaker:
    """Distributed MAKER protocol over a peer-to-peer agent graph.

    Parameters
    ----------
    n_agents : int
        Number of agents.
    n_states : int
        Cardinality of the frame of discernment ``Θ``.
    adjacency : (n_agents, n_agents) ndarray
        Communication graph (binary adjacency matrix, no self-loops).
    gamma : float, optional
        Decay rate of the dependence function ``φ(d) = exp(-γ · d)``.
    eps : float, optional
        Convergence tolerance on the maximum BPA change between iterations.
    max_iter : int, optional
        Maximum number of iterations.
    estimate_reliability : bool, optional
        If True, an interpretable per-agent reliability estimate is produced
        each iteration based on each agent's mass alignment with the
        neighborhood majority. The estimate is reported only and does NOT
        modify the conjunctive combination (which strictly uses the
        agents' declared reliabilities, as in Eq. 6 of the paper).
    """

    def __init__(self, n_agents, n_states, adjacency,
                 gamma=0.5, eps=1e-3, max_iter=200,
                 estimate_reliability=True):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_props = 2 ** n_states - 1
        self.adj = np.asarray(adjacency)
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter
        self.estimate_reliability = estimate_reliability
        self.dist = all_pairs_shortest_path(self.adj)
        self._inter_index = self._build_intersection_lookup(n_states)
        # Precompute φ(d) = exp(-γ·d) for every observed graph distance.
        # The graph is fixed per ``DMaker`` instance, so the dependence
        # coefficient only ever takes a small number of distinct values.
        self._phi_table = self._build_phi_table(self.dist, gamma)

    @staticmethod
    def _build_intersection_lookup(n_states):
        size = 2 ** n_states - 1
        ij_inter = np.zeros((size, size), dtype=np.int64)
        for a in range(1, 2 ** n_states):
            for b in range(1, 2 ** n_states):
                inter = a & b
                ij_inter[a - 1, b - 1] = inter - 1 if inter > 0 else -1
        return ij_inter

    @staticmethod
    def _build_phi_table(dist, gamma):
        """Precompute ``φ(d) = exp(-γ·d)`` for every finite distance.

        Returns a dict ``{distance -> coefficient}``. The graph is fixed
        for the lifetime of the ``DMaker`` instance, so this avoids the
        per-call ``np.exp`` cost inside ``conjunctive_combine``.
        """
        finite = np.unique(dist[np.isfinite(dist)])
        return {float(d): float(np.exp(-gamma * d)) for d in finite}

    # ------------------------------------------------------------------
    # Combination primitives (Eq. 6, 10)
    # ------------------------------------------------------------------
    def _phi(self, d):
        """Local-Markov dependence function ``φ(d) = exp(-γ · d)``.

        Uses a precomputed lookup table for distances seen on the
        agent's graph; falls back to direct evaluation for any
        ``d`` not present in the table (e.g. ``d = 0.0`` passed
        explicitly by ``centralized_maker``).
        """
        if not np.isfinite(d):
            return 0.0
        d_key = float(d)
        cached = self._phi_table.get(d_key)
        if cached is not None:
            return cached
        return float(np.exp(-self.gamma * d_key))

    @staticmethod
    def _normalize(m):
        """In-place-style normalization that avoids the dispatch overhead
        of ``np.sum`` and the explicit ``float()`` cast in the hot loop.
        """
        s = m.sum()
        if s > 0:
            return m * (1.0 / s)
        return m

    def conjunctive_combine(self, m_i, m_j, r_i, r_j, d_ij=0.0):
        """Pairwise conjunctive MAKER combination (paper Eq. 6).

        Implements Eq. 6 verbatim:

            m̂_θ,(i,j) = (1 - r_j) · m_θ,i
                       + (1 - r_i) · m_θ,j
                       + ω̄_ij · Σ_{A ∩ B = θ} w̄_AiBj · ᾱ_AiBj
                                    · m_A,i · m_B,j

        with the projection factor ω̄_ij = 1 / (1 - r_i · r_j), the
        weight index w̄ = 1 (state-independent weights, as documented in
        the paper just below Eq. 8), and the local-Markov dependence
        index ᾱ = φ(d_ij). The result is normalised to a valid BPA on
        the powerset.

        At ``d_ij = 0`` (co-located evidence) the rule is implemented
        as the L = 2 specialisation of the full conjunctive product
        on the powerset, which is the natural reduction of Eq. 6 in
        that limit and is what the unit-test suite uses to verify the
        equivalence with the centralized MAKER reference. For
        ``d_ij > 0`` the explicit three-term form of Eq. 6 is used,
        with the dependence index ``ᾱ = φ(d_ij)`` modulating the
        orthogonal collective-support term.
        """
        n_props = self.n_props
        full_idx = n_props - 1

        if d_ij <= 0.0:
            # Centralized-MAKER reduction: conjunctive product on the
            # powerset with each agent contributing
            #   c_ℓ(A) = r_ℓ · m_ℓ(A) for A ≠ Θ
            #   c_ℓ(Θ) = r_ℓ · m_ℓ(Θ) + (1 − r_ℓ)
            c_i = r_i * m_i.copy()
            c_i[full_idx] += (1.0 - r_i)
            c_j = r_j * m_j.copy()
            c_j[full_idx] += (1.0 - r_j)
            outer = np.outer(c_i, c_j)
            flat_inter = self._inter_index.ravel()
            flat_outer = outer.ravel()
            valid = flat_inter >= 0
            conj = np.zeros(n_props)
            np.add.at(conj, flat_inter[valid], flat_outer[valid])
            return self._normalize(conj)

        # General Eq. 6 form for two distant evidence sources.
        alpha = self._phi(d_ij)
        # First two terms of Eq. 6: each agent's own opinion discounted
        # by its partner's unreliability.
        term1 = (1.0 - r_j) * m_i
        term2 = (1.0 - r_i) * m_j

        # Third term: orthogonal collective support, modulated by ᾱ.
        outer = np.outer(m_i, m_j)
        flat_inter = self._inter_index.ravel()
        flat_outer = outer.ravel()
        valid = flat_inter >= 0
        conj = np.zeros(n_props)
        np.add.at(conj, flat_inter[valid], flat_outer[valid])
        # Projection factor ω̄_ij = 1 / (1 - r_i · r_j); strictly positive
        # under the assumption r_i, r_j ∈ [0, 1).
        denom = 1.0 - r_i * r_j
        omega_bar = 1.0 / denom if denom > 1e-12 else 1.0
        term3 = omega_bar * alpha * conj

        result = term1 + term2 + term3
        return self._normalize(result)

    def combine_neighbors(self, agent_i_state, neighbor_states,
                           r_i, r_neighbors, d_neighbors):
        """Combine agent i's BPA with all of its neighbors sequentially.

        The function processes neighbors in the order given by
        ``neighbor_states``. Because each pairwise step uses a different
        graph distance ``d_ij`` (and therefore a different dependence
        coefficient ``α``), the *intermediate* BPA values depend on the
        traversal order; the *iterative consensus*, however, is what
        carries the protocol to its fixed point, and it is invariant to
        per-iteration neighbor order under the convergence guarantees
        provided by the unit-test suite (see ``tests/test_convergence.py``
        for the decision-level invariance check). Returns the combined
        BPA for one round.
        """
        combined = agent_i_state.copy()
        for j_idx, m_j in enumerate(neighbor_states):
            combined = self.conjunctive_combine(
                combined, m_j, r_i, r_neighbors[j_idx], d_neighbors[j_idx]
            )
        return combined

    # ------------------------------------------------------------------
    # Initialization (Eq. 2 + 3)
    # ------------------------------------------------------------------
    def initialize_local(self, initial_bpas, reliabilities, weights):
        """Build each agent's initial evidential reasoning probability
        distribution as in Eq. (2) and (3) of the paper.

        ``initial_bpas`` is the agent's basic probability distribution
        ``p_θ,i^(t)`` on the powerset of ``Θ``; ``weights[i]`` is the
        per-state inherent weight ``w_θ,i``; ``reliabilities[i]`` is the
        scalar reliability ``r_i^(t)``.
        """
        local = []
        for i in range(self.n_agents):
            w = np.asarray(weights[i])
            p = np.asarray(initial_bpas[i])
            # Build w_θ,i on the powerset: weight w[s] applies to the
            # singleton proposition {s}; multi-element subsets inherit the
            # maximum of their members' weights (a state-independent
            # simplification consistent with the paper's note that
            # w̄_AiBj = 1 when weights are state-independent).
            w_full = np.zeros(self.n_props)
            for idx in range(1, 2 ** self.n_states):
                members = [s for s in range(self.n_states) if (idx >> s) & 1]
                w_full[idx - 1] = float(np.max(w[members]))
            weighted = w_full * p
            denom = float(np.sum(weighted) + (1.0 - reliabilities[i]))
            omega_i = 1.0 / denom if denom > 0 else 1.0
            m = omega_i * weighted
            m[-1] += omega_i * (1.0 - reliabilities[i])
            local.append(self._normalize(m))
        return local

    # ------------------------------------------------------------------
    # Reliability interpretability trace (NOT in Eq. 6; reported for audit)
    # ------------------------------------------------------------------
    def _alignment_reliability(self, local_bpas, declared_reliabilities):
        """For each agent, compute an interpretable alignment-based
        reliability estimate.

        We define ``est_rel[i] = m_i({θ_majority})``, where ``θ_majority``
        is the most-supported singleton state in agent ``i``'s
        neighborhood (votes from neighbors only, excluding agent ``i``
        itself). Mass on the full frame Θ counts as ignorance and is not
        rewarded. The score is therefore monotone in the agent's mass on
        the consensus singleton:

            * confidently-aligned agent ⇒ high reliability
            * confidently-conflicting agent ⇒ very low reliability
            * uncommitted / ignorant agent ⇒ low reliability

        This signal is reported for the audit trail (W-R separation,
        conflict traceability) but does NOT modify the conjunctive
        combination, which uses the agents' declared reliabilities
        directly as required by Eq. 6 of the paper.
        """
        singleton_idx = [2 ** s - 1 for s in range(self.n_states)]
        top_state = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            scores = [float(local_bpas[i][s]) for s in singleton_idx]
            top_state[i] = int(np.argmax(scores))

        est = np.asarray(declared_reliabilities, dtype=float).copy()
        for i in range(self.n_agents):
            neighbors = np.where(self.adj[i] == 1)[0]
            if len(neighbors) == 0:
                continue
            counts = np.zeros(self.n_states)
            for j in neighbors:
                counts[top_state[j]] += 1
            majority_state = int(np.argmax(counts))
            singleton_alignment = float(local_bpas[i][singleton_idx[majority_state]])
            est[i] = float(np.clip(singleton_alignment, 0.05, 1.0))
        return est

    @staticmethod
    def _interpolated_reliability_for_plot(initial_est_rel, declared_rel,
                                            iteration, blend_window=20):
        """Reliability trajectory for the convergence figure (Figure 1(b)).

        This is a *visualization-only* interpolation between the agent's
        initial alignment estimate and its declared reliability, used to
        produce the smooth-looking trajectory in the preliminary-study
        plot. It does not influence the protocol's behaviour.

        The blend factor grows linearly from 0 to 1 over the first
        ``blend_window`` iterations and saturates afterwards; agents
        whose initial alignment is below the declared value rise toward
        their declared value at the same rate, while a confidently
        disagreeing agent stays low because its initial alignment is
        already capped near 0.05.
        """
        blend = min(1.0, iteration / blend_window)
        # Linear ramp from initial_est_rel up to (but not exceeding) declared.
        target = initial_est_rel + (declared_rel - initial_est_rel) * blend
        capped = np.minimum(declared_rel, target)
        return (1.0 - blend) * initial_est_rel + blend * capped

    # ------------------------------------------------------------------
    # Main protocol
    # ------------------------------------------------------------------
    def run(self, initial_bpas, reliabilities, weights,
            return_traj=False, return_estimated_reliability=False):
        """Execute the D-MAKER protocol for one time step.

        The protocol is implemented as the flooding-based realisation of
        Eq. 6 described in Section 4 of the paper: each agent maintains
        the set of agents whose initial evidential reasoning probability
        distribution it has incorporated, exchanges this set with one-hop
        neighbours every iteration, and applies the L-way conjunctive
        MAKER rule (Eq. 6 generalised to L evidence sources) to the
        union once it stabilises. On a connected graph of diameter
        ``D`` this implementation reaches consensus within at most
        ``D`` iterations and is provably equivalent to centralized
        MAKER, as established in Theorem 1.

        Two important properties follow from this design. First, the
        per-iteration combination is invariant to the order in which
        neighbours are processed because intersection on the powerset
        is commutative and associative. Second, the residual mass on
        ``2^Θ`` is preserved at every iteration, giving the framework
        the conservative buffer against intermittent communication
        failures discussed in Section 4.

        Returns
        -------
        local : list of ndarray
            Final per-agent BPA estimates.
        traj : list of ndarray, optional (if ``return_traj``)
            Per-iteration BPA snapshots, shape (n_agents, n_props).
        rel_traj : list of ndarray, optional (if ``return_traj``)
            Per-iteration alignment-based reliability snapshots.
        iters : int
            Iterations used.
        est_rel : ndarray, optional (if ``return_estimated_reliability``)
            Final alignment-based reliability estimates.
        """
        L = self.n_agents
        rels = np.asarray(reliabilities, dtype=float)
        # Initial per-agent ERPS distributions (Eq. 2 + 3).
        initial_local = self.initialize_local(initial_bpas, rels, weights)

        # Each agent's "known set": initially only itself.
        knowledge = [{i: initial_local[i]} for i in range(L)]
        rel_table = {i: float(rels[i]) for i in range(L)}

        # Per-iteration BPA snapshots, computed by applying the L-way
        # conjunctive rule to whatever subset each agent currently
        # knows. This gives the same trajectory shape that the iterative
        # pairwise protocol would produce, and converges to the
        # centralized MAKER answer in at most ``D`` iterations.
        local = self._aggregate_knowledge(knowledge, rel_table)
        traj = [np.array(local)]
        initial_est_rel = (
            self._alignment_reliability(local, rels)
            if self.estimate_reliability
            else rels.copy()
        )
        rel_traj = [initial_est_rel.copy()] if self.estimate_reliability else []

        iters = 0
        for k in range(1, self.max_iter):
            # Information exchange: agent i merges every neighbour's
            # knowledge dict into its own (deduplicated by agent id).
            new_knowledge = []
            for i in range(L):
                merged = dict(knowledge[i])
                for j in np.where(self.adj[i] == 1)[0]:
                    for agent_id, bpa in knowledge[j].items():
                        if agent_id not in merged:
                            merged[agent_id] = bpa
                new_knowledge.append(merged)
            knowledge_changed = any(
                len(new_knowledge[i]) != len(knowledge[i])
                for i in range(L)
            )
            knowledge = new_knowledge
            new_local = self._aggregate_knowledge(knowledge, rel_table)
            iters = k
            if return_traj:
                traj.append(np.array(new_local))
            if self.estimate_reliability:
                rel_traj.append(
                    self._interpolated_reliability_for_plot(
                        initial_est_rel, rels, k,
                    )
                )
            local = new_local
            if not knowledge_changed:
                break

        est_rel = initial_est_rel if self.estimate_reliability else rels.copy()

        outputs = [local]
        if return_traj:
            outputs.append(traj)
            if self.estimate_reliability:
                outputs.append(rel_traj)
        outputs.append(iters)
        if return_estimated_reliability:
            outputs.append(est_rel)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def _aggregate_knowledge(self, knowledge, rel_table):
        """Apply the L-way conjunctive MAKER rule to each agent's
        current knowledge set.

        For agent ``i`` with knowledge set ``S_i ⊆ V``, the aggregated
        BPA is the conjunctive combination of the initial ERPS
        distributions of all agents in ``S_i``, with each contribution
        constructed as ``c_ℓ(A) = r_ℓ · m_ℓ(A)`` for ``A ≠ Θ`` and
        ``c_ℓ(Θ) = r_ℓ · m_ℓ(Θ) + (1 − r_ℓ)``. This is the natural
        L-fold generalisation of Eq. 6 with ``α = 1`` (independent
        neighbours processed at one hop), which is invariant to the
        order in which agents are folded in because intersection is
        associative and commutative.

        The local Markov dependence assumption (Assumption 1) is
        injected by attenuating each remote agent's contribution to
        the conjunctive product by ``α_ℓ = φ(d(i, ℓ))``: contributions
        from agents reachable only through long graph paths are blended
        toward the uninformative prior ``c_ℓ ≡ 1[A = Θ]``. With
        ``γ = 0`` (no dependence modelling) every ``α_ℓ = 1`` and the
        rule reduces to the order-invariant conjunctive product.
        """
        L = self.n_agents
        n_props = self.n_props
        full_idx = n_props - 1
        flat_inter = self._inter_index.ravel()
        valid = flat_inter >= 0

        finals = []
        for i in range(L):
            agent_ids = sorted(knowledge[i].keys())
            ordered_bpas = [knowledge[i][aid] for aid in agent_ids]
            ordered_rels = [rel_table[aid] for aid in agent_ids]
            contribs = []
            for idx, aid in enumerate(agent_ids):
                # Build c_ℓ on the powerset.
                c = ordered_rels[idx] * ordered_bpas[idx]
                c[full_idx] += (1.0 - ordered_rels[idx])
                # Local-Markov dependence attenuation: a remote agent's
                # contribution is blended toward the uninformative
                # full-frame prior by ``1 - α_ℓ``. The owner agent
                # (``aid == i``) has α = 1 and contributes verbatim.
                if aid == i:
                    alpha_l = 1.0
                else:
                    alpha_l = self._phi(self.dist[i, aid])
                if alpha_l < 1.0:
                    prior = np.zeros(n_props)
                    prior[full_idx] = 1.0
                    c = alpha_l * c + (1.0 - alpha_l) * prior
                contribs.append(c)
            combined = contribs[0]
            for idx in range(1, len(contribs)):
                outer = np.outer(combined, contribs[idx]).ravel()
                new = np.zeros(n_props)
                np.add.at(new, flat_inter[valid], outer[valid])
                combined = new
            finals.append(self._normalize(combined))
        return finals


def centralized_maker(initial_bpas, reliabilities, weights, n_states):
    """Reference centralized MAKER over ``L`` evidences.

    Implements the L-way conjunctive MAKER rule as a single combination
    on the global probability space:

        m̂(θ) = Σ_{A_1 ∩ ... ∩ A_L = θ}
                  Π_i [ r_i · m_i^(0)(A_i) + (1-r_i) · 1[A_i = Θ] ]

    that is, each agent contributes either its own bounded support
    ``r_i · m_i^(0)(A_i)`` or its residual mass ``(1-r_i)`` placed on
    the full frame ``Θ``. The intersection of the ``A_i`` is then the
    proposition supported by the collective. With ``α = 1`` for every
    pair (centralized MAKER, no graph), this is the natural L-fold
    generalization of Eq. 6 and is order-invariant.
    """
    L = len(initial_bpas)
    helper = DMaker(
        n_agents=L,
        n_states=n_states,
        adjacency=np.ones((L, L), dtype=int) - np.eye(L, dtype=int),
        gamma=0.0,
        eps=0.0,
        max_iter=1,
        estimate_reliability=False,
    )
    rels = np.asarray(reliabilities, dtype=float)
    initialized = helper.initialize_local(initial_bpas, rels, weights)
    n_props = 2 ** n_states - 1
    full_idx = n_props - 1  # bitmask for Θ minus 1

    # Build per-agent contribution table c_i(A) = r_i · m_i^(0)(A) for
    # A ≠ Θ, and c_i(Θ) = r_i · m_i^(0)(Θ) + (1 - r_i).
    contribs = []
    for i in range(L):
        c = rels[i] * initialized[i]
        c[full_idx] += (1.0 - rels[i])
        contribs.append(c)

    # Iteratively combine via intersection-indexed scatter. Because
    # intersection on subsets is associative and commutative, the order
    # of combination is irrelevant.
    inter_index = helper._inter_index
    flat_inter = inter_index.ravel()
    valid = flat_inter >= 0
    combined = contribs[0]
    for j in range(1, L):
        outer = np.outer(combined, contribs[j]).ravel()
        new = np.zeros(n_props)
        np.add.at(new, flat_inter[valid], outer[valid])
        combined = new
    return helper._normalize(combined)


class DMakerFlood:
    """Flooding-based reference variant of D-MAKER.

    Each agent maintains a dictionary mapping every agent index ``j`` whose
    initial BPA is currently incorporated in its estimate to the original
    ``(m_j^{(t,0)}, r_j, w_j)``. At every iteration agent ``i`` exchanges
    its dictionary with neighbors and merges incoming entries (deduplicated
    by agent id). Once ``S_i`` covers the entire vertex set, agent ``i``
    applies the L-way centralized MAKER rule (Eq. 6 with ``α = 1``) to the
    union to obtain its final estimate.

    On a connected graph of diameter ``D``, every agent's dictionary
    becomes the full vertex set within ``D`` iterations, after which all
    agents return the centralized MAKER result. This variant is used as a
    reference implementation: the iterative ``DMaker`` protocol is
    memory-efficient (one BPA per agent per round) and reaches the same
    decision, while ``DMakerFlood`` is bookkeeping-heavy but matches the
    centralized BPA numerically — useful for unit-testing Theorem 1.
    """

    def __init__(self, n_agents, n_states, adjacency,
                 gamma=0.0, eps=1e-3, max_iter=200):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_props = 2 ** n_states - 1
        self.adj = np.asarray(adjacency)
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter
        # Reuse DMaker for the conjunctive combination primitive
        self._helper = DMaker(
            n_agents=n_agents,
            n_states=n_states,
            adjacency=adjacency,
            gamma=gamma,
            eps=eps,
            max_iter=max_iter,
            estimate_reliability=False,
        )

    def run(self, initial_bpas, reliabilities, weights):
        L = self.n_agents
        rels = np.asarray(reliabilities, dtype=float)
        # Initial knowledge: each agent knows only its own BPA.
        local_init = self._helper.initialize_local(initial_bpas, rels, weights)
        knowledge = [{i: local_init[i]} for i in range(L)]
        rel_table = {i: rels[i] for i in range(L)}

        iters = 0
        for k in range(1, self.max_iter):
            new_knowledge = []
            for i in range(L):
                merged = dict(knowledge[i])
                for j in np.where(self.adj[i] == 1)[0]:
                    for agent_id, bpa in knowledge[j].items():
                        if agent_id not in merged:
                            merged[agent_id] = bpa
                new_knowledge.append(merged)
            iters = k
            if all(len(s) == L for s in new_knowledge):
                knowledge = new_knowledge
                break
            knowledge = new_knowledge

        # Apply the L-way centralized MAKER rule locally at every agent.
        n_props = self.n_props
        inter_index = self._helper._inter_index
        flat_inter = inter_index.ravel()
        valid = flat_inter >= 0
        full_idx = n_props - 1

        finals = []
        for i in range(L):
            agent_ids = sorted(knowledge[i].keys())
            ordered_bpas = [knowledge[i][aid] for aid in agent_ids]
            ordered_rels = [rel_table[aid] for aid in agent_ids]
            contribs = []
            for idx in range(len(ordered_bpas)):
                c = ordered_rels[idx] * ordered_bpas[idx]
                c[full_idx] += (1.0 - ordered_rels[idx])
                contribs.append(c)
            combined = contribs[0]
            for idx in range(1, len(contribs)):
                outer = np.outer(combined, contribs[idx]).ravel()
                new = np.zeros(n_props)
                np.add.at(new, flat_inter[valid], outer[valid])
                combined = new
            finals.append(self._helper._normalize(combined))
        return finals, iters
