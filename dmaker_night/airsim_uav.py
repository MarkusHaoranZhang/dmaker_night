"""AirSim-based UAV swarm dataset loader.

This module loads the BPA evidence file produced by the AirSim + ResNet-18
pipeline used in Section 5.1.1 of the paper. Six UAVs observe the same
ground vehicle from different viewing angles and distances inside the
AirSim simulation platform; each UAV's onboard ResNet-18 classifier (fine-
tuned with a different noise level) produces a softmax probability vector
that is converted to a basic probability assignment on the powerset of
the target frame ``Θ = {friend, foe, neutral}``. Two of the six UAVs are
subject to artificial communication jamming, modeled as a per-step
probability of degraded reliability.

The pre-computed BPA tensor is shipped under ``data/airsim_uav_bpa.npz``
so that downstream experiments do not require an AirSim environment or a
GPU. The loader exposes the same trial-dict contract as
:class:`dmaker_night.data.UAVSwarmDataset`, allowing it to be substituted in any
existing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .data import Trial


def _data_path() -> Path:
    """Resolve the bundled ``airsim_uav_bpa.npz`` path."""
    return Path(__file__).resolve().parents[1] / "data" / "airsim_uav_bpa.npz"


class AirSimUAVDataset:
    """AirSim-based UAV swarm target identification dataset.

    Loads the BPA evidence pre-computed from AirSim multi-view image
    sequences and ResNet-18 inference. The pipeline that produced
    ``airsim_uav_bpa.npz`` follows the description in Section 5.1.1 of
    the paper:

    - 6 UAVs observe ground targets from different angles and distances
    - Communication topology follows physical proximity
    - Each UAV carries a ResNet-18 classifier fine-tuned with a different
      additive noise level (σ ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25})
    - UAVs 4 and 5 are subject to intermittent jamming (per-step
      probability of degraded reliability)
    - Each ResNet-18 softmax output is converted to a BPA on the powerset
      of ``Θ`` using reliability-weighted singleton allocation with the
      residual mass placed on the full frame ``Θ``

    Parameters
    ----------
    n_trials : int, optional
        Number of trials to expose. The bundled file contains 200 trials;
        smaller values truncate the list (after a deterministic shuffle
        controlled by ``seed``).
    seed : int, optional
        Determines the deterministic reshuffling and subset selection
        when ``n_trials`` is smaller than the bundled trial count.
    path : str or Path, optional
        Override path to the BPA file. Defaults to
        ``<repo>/data/airsim_uav_bpa.npz``.
    """

    n_agents: int = 6
    n_states: int = 3
    n_timesteps: int = 15

    def __init__(self, n_trials: int = 40, seed: int = 456,
                 path: str | Path | None = None):
        self.n_trials = n_trials
        self.seed = seed
        self._path = Path(path) if path is not None else _data_path()
        if not self._path.exists():
            raise FileNotFoundError(
                f"AirSim UAV BPA file not found at {self._path}. "
                "Run ``python -m dmaker_night.build_airsim_uav`` to regenerate it."
            )

    def generate(self) -> List[Trial]:
        """Load and return the trial list from the bundled BPA file.

        ``seed`` deterministically reshuffles which subset of the cached
        trials is exposed and the order in which they appear, so that
        different seeds yield different but reproducible trial lists.
        This matches the behaviour of the procedural simulators in
        :mod:`dmaker_night.data` and lets the experiment harness average over
        independent realisations.
        """
        with np.load(self._path, allow_pickle=True) as data:
            evidence_arr = data["evidence"]            # (T, A, S, P)
            reliabilities_arr = data["reliabilities"]  # (T, S, A)
            weights_arr = data["weights"]              # (T, A, n_states)
            true_states_arr = data["true_states"]       # (T,)

        n_available = int(evidence_arr.shape[0])
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n_available)
        n = min(self.n_trials, n_available)
        order = order[:n]

        trials: List[Trial] = []
        for t in order:
            t = int(t)
            evidence_list = [
                [evidence_arr[t, a, s] for s in range(evidence_arr.shape[2])]
                for a in range(evidence_arr.shape[1])
            ]
            trials.append({
                "true_state": int(true_states_arr[t]),
                "evidence": evidence_list,
                "reliabilities": np.asarray(reliabilities_arr[t], dtype=float),
                "weights": [
                    np.asarray(weights_arr[t, a], dtype=float)
                    for a in range(weights_arr.shape[1])
                ],
            })
        return trials
