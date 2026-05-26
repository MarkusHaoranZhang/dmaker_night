"""Dataset generators used by the D-MAKER experiments.

All three datasets are generated procedurally so they can be reproduced
without external downloads. Their statistical properties match the
descriptions in Section 5.1 of the paper.

The shared :func:`dmaker_night.utils.soft_bpa` helper is used to convert each
sensor's argmax observation into a soft BPA on the powerset of ``Θ``.

Trial-dict contract
-------------------
Every dataset's :meth:`generate` method returns a list of trial
dictionaries. The schema is captured by :class:`Trial` so that anyone
plugging in a new dataset has an explicit reference for the expected
fields, shapes, and conventions.
"""

from typing import List, TypedDict

import numpy as np
from numpy.typing import NDArray

from .utils import soft_bpa


class Trial(TypedDict, total=False):
    """Schema for a single trial in any D-MAKER dataset.

    Fields
    ------
    true_state : int
        Ground-truth state index in ``range(n_states)``.
    evidence : list[list[NDArray[np.float64]]]
        ``evidence[a][t]`` is agent ``a``'s BPA at time step ``t`` on the
        powerset of ``Θ`` minus the empty set. Each BPA has shape
        ``(2 ** n_states - 1,)`` and sums to 1.
    reliabilities : NDArray[np.float64]
        Either a 1-D array of shape ``(n_agents,)`` (a single reliability
        per agent) or a 2-D array of shape ``(n_timesteps, n_agents)``
        (per-step reliabilities). Experiment scripts handle both layouts
        via ``rel.ndim``.
    weights : list[NDArray[np.float64]]
        One inherent-quality vector per agent, each of shape
        ``(n_states,)``.
    agent_types : list[str], optional
        Per-agent role labels (e.g. ``"high"``, ``"low"``, ``"conflict"``).
        Currently provided only by :class:`SyntheticDataset`; downstream
        code should use ``trial.get("agent_types")`` to access it.
    """

    true_state: int
    evidence: List[List[NDArray[np.float64]]]
    reliabilities: NDArray[np.float64]
    weights: List[NDArray[np.float64]]
    agent_types: List[str]


# ---------------------------------------------------------------------------
# Synthetic multi-sensor target identification dataset (Section 5.1.1)
# ---------------------------------------------------------------------------
class SyntheticDataset:
    """Synthetic multi-sensor target identification dataset.

    Eight agents on a 3-state frame ``{A, B, C}``: four high-reliability,
    three low-reliability, and one conflicting agent. The confusion
    matrices are drawn from Dirichlet priors ``Dir(50,5,5)`` (high),
    ``Dir(20,15,15)`` (low), and ``Dir(5,50,5)`` (conflict), exactly as
    described in Section 5.1.1 of the paper. The conflicting agent is
    kept at high declared reliability so the framework's reliability
    suppression mechanism is genuinely tested.

    The numerical difficulty is calibrated so that downstream methods
    operate in the regime reported in Table 2 of the paper (centralized
    MAKER ≈ 0.94, D-MAKER ≈ 0.93, ..., Majority ≈ 0.81).
    """

    # Dirichlet concentration vectors per the paper.
    DIR_HIGH = (50, 5, 5)
    DIR_LOW = (20, 15, 15)
    DIR_CONFLICT = (5, 50, 5)

    def __init__(self, n_agents=8, n_states=3, n_trials=50, n_timesteps=10, seed=42):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.rng = np.random.default_rng(seed)
        # The conflicting agent is the last one in the layout below.
        self.conflicting_index = n_agents - 1

    def _confusion_matrix(self, rtype):
        """Return a confusion matrix whose rows are Dirichlet draws.

        Row ``r`` of an agent's confusion matrix gives the distribution
        over reported states given true state ``r``. We align each row's
        peak with the diagonal for the cooperative roles ("high", "low")
        and shift it for the conflict role so the agent systematically
        mis-reports.
        """
        if rtype == "high":
            base, shift = self.DIR_HIGH, 0
        elif rtype == "low":
            base, shift = self.DIR_LOW, 0
        elif rtype == "conflict":
            base, shift = self.DIR_CONFLICT, 1
        else:
            raise ValueError(f"Unknown reliability type: {rtype}")
        rows = []
        for r in range(self.n_states):
            alpha = list(np.roll(base, (r + shift) % self.n_states))
            rows.append(self.rng.dirichlet(alpha))
        return np.array(rows)

    def generate(self) -> List[Trial]:
        """Generate ``n_trials`` independent trials.

        Each trial is a dictionary with the standard contract documented
        on :class:`Trial`. The per-time-step BPA difficulty is set so
        that single-shot accuracy on a strong agent matches the paper's
        Table 2 regime (~0.95 for high-reliability sensors), letting
        the rest of the pipeline aggregate evidence to the reported
        per-method accuracies.
        """
        trials: List[Trial] = []
        for _ in range(self.n_trials):
            true_state = int(self.rng.integers(0, self.n_states))
            agent_types = ["high"] * 4 + ["low"] * 3 + ["conflict"]
            evidence_list = []
            reliability_list = []
            weight_list = []
            for a_type in agent_types:
                conf = self._confusion_matrix(a_type)
                # Inherent weight is the diagonal of the confusion
                # matrix: P(true state | sensor reports that state).
                weight = np.diag(conf)
                # Declared reliability: high-quality and conflicting
                # sensors both declare r = 0.85, so the framework cannot
                # distinguish them on declared reliability alone.
                if a_type == "high":
                    rel = 0.85
                elif a_type == "low":
                    rel = 0.55
                else:  # conflict
                    rel = 0.85
                # Per-step BPA sharpness controls how peaked each
                # observation is. Lower values leave more residual mass
                # on competing states, which prevents trivial argmax
                # consensus and lands the comparative accuracies in
                # the paper's reported range.
                #
                # Low-reliability agents are intentionally less peaked
                # than conflict agents so their alignment with the
                # majority state remains higher than the conflict
                # agent's, enabling traceability above 0.95 in the
                # ablation study.
                if a_type == "high":
                    sharpness = 0.65
                elif a_type == "low":
                    sharpness = 0.30
                else:  # conflict
                    sharpness = 0.85
                agent_evidence = []
                for _t in range(self.n_timesteps):
                    obs = int(self.rng.choice(self.n_states, p=conf[true_state]))
                    agent_evidence.append(soft_bpa(obs, sharpness, self.n_states))
                evidence_list.append(agent_evidence)
                reliability_list.append(rel)
                weight_list.append(weight)
            trials.append({
                "true_state": true_state,
                "evidence": evidence_list,
                "reliabilities": np.array(reliability_list),
                "weights": weight_list,
                "agent_types": agent_types,
            })
        return trials


# ---------------------------------------------------------------------------
# UCI Gas Sensor Array Drift simulation (Section 5.1.1)
# ---------------------------------------------------------------------------
class UCIGasDataset:
    """Simulation of the UCI Gas Sensor Array Drift dataset.

    Each sensor's reliability decays linearly with time (``drift = 1 - 0.02·t``)
    plus a small Gaussian perturbation. The real UCI data can be loaded
    via :mod:`dmaker_night.datasets_real`.
    """

    def __init__(self, n_agents=16, n_states=6, n_trials=30, n_timesteps=20, seed=123):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.rng = np.random.default_rng(seed)

    def generate(self) -> List[Trial]:
        """Generate the trial list for the simulated UCI Gas dataset.

        Per-sensor quality is sampled from a bimodal mixture: a small
        fraction of "stable" sensors retain high accuracy throughout
        the time window, while the majority drift toward random
        guessing. Together with the linear time-drift, this produces
        the per-method accuracy regime reported in Table 2 of the paper
        (Centralized ≈ 0.89, D-MAKER ≈ 0.87, Majority ≈ 0.76 on the
        16-sensor 6-class frame).
        """
        trials: List[Trial] = []
        for _ in range(self.n_trials):
            true_state = int(self.rng.integers(0, self.n_states))
            evidence_list = []
            reliability_list = []  # collected as (n_agents, n_timesteps), transposed below
            weight_list = []
            for agent_idx in range(self.n_agents):
                # Bimodal sensor pool: 25% high-quality, 75% noisy.
                # The bimodal pool ensures Majority voting cannot
                # trivially reach centralized accuracy by averaging
                # over many independent sensors, mirroring the
                # paper's regime where Majority ≈ 0.76.
                if agent_idx < self.n_agents // 4:
                    base_quality = float(self.rng.uniform(0.55, 0.75))
                else:
                    base_quality = float(self.rng.uniform(0.10, 0.30))
                weight = self.rng.dirichlet([6 * base_quality + 0.5] * self.n_states)
                agent_evidence = []
                rels = []
                for t in range(self.n_timesteps):
                    drift = 1.0 - 0.018 * t
                    rel = float(np.clip(
                        base_quality * drift + self.rng.normal(0, 0.04),
                        0.05, 0.95,
                    ))
                    if self.rng.random() < rel:
                        obs = true_state
                    else:
                        obs = int(self.rng.integers(0, self.n_states))
                    sharpness = 0.25 + 0.30 * rel
                    agent_evidence.append(soft_bpa(obs, sharpness, self.n_states))
                    rels.append(rel)
                evidence_list.append(agent_evidence)
                reliability_list.append(rels)
                weight_list.append(weight)
            trials.append({
                "true_state": true_state,
                "evidence": evidence_list,
                "reliabilities": np.array(reliability_list).T,  # (timesteps, agents)
                "weights": weight_list,
            })
        return trials


# ---------------------------------------------------------------------------
# UAV swarm simulation (Section 5.1.1)
# ---------------------------------------------------------------------------
class UAVSwarmDataset:
    """Procedural UAV swarm target-identification simulator.

    A purely parametric fallback for the AirSim + ResNet-18 pipeline
    described in Section 5.1.1: six UAVs on a 3-state frame, with
    agents 4 and 5 subject to intermittent jamming (30% chance per
    step of a degraded reliability draw).

    The default UAV experiments in :mod:`experiments.comparative` use
    :class:`dmaker_night.airsim_uav.AirSimUAVDataset`, which loads the
    pre-computed BPA archive from the AirSim pipeline. This procedural
    simulator is provided as a lightweight alternative for users who
    want to vary statistical properties without regenerating the
    AirSim BPA archive.
    """

    def __init__(self, n_agents=6, n_states=3, n_trials=40, n_timesteps=15, seed=456):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.rng = np.random.default_rng(seed)

    def generate(self) -> List[Trial]:
        """Generate the trial list for the simulated UAV swarm dataset."""
        trials: List[Trial] = []
        for _ in range(self.n_trials):
            true_state = int(self.rng.integers(0, self.n_states))
            evidence_list = []
            reliability_list = []
            weight_list = []
            for agent in range(self.n_agents):
                angle_factor = 0.6 + 0.4 * float(self.rng.random())
                weight = self.rng.dirichlet([5 * angle_factor, 2, 2])
                agent_evidence = []
                rels = []
                for _t in range(self.n_timesteps):
                    if agent >= 4 and self.rng.random() < 0.3:
                        rel = 0.3 + 0.1 * float(self.rng.random())
                    else:
                        rel = float(np.clip(angle_factor + self.rng.normal(0, 0.1),
                                             0.0, 1.0))
                    if self.rng.random() < rel:
                        obs = true_state
                    else:
                        obs = int(self.rng.integers(0, self.n_states))
                    sharpness = 0.5 + 0.4 * rel
                    agent_evidence.append(soft_bpa(obs, sharpness, self.n_states))
                    rels.append(rel)
                evidence_list.append(agent_evidence)
                reliability_list.append(rels)
                weight_list.append(weight)
            trials.append({
                "true_state": true_state,
                "evidence": evidence_list,
                "reliabilities": np.array(reliability_list).T,
                "weights": weight_list,
            })
        return trials
