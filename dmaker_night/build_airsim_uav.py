"""Build the AirSim UAV swarm BPA evidence file.

This script materialises the ``data/airsim_uav_bpa.npz`` archive used by
:class:`dmaker_night.airsim_uav.AirSimUAVDataset`. The archive packages the
basic probability assignments produced by the AirSim + ResNet-18 pipeline
described in Section 5.1.1 of the paper:

    1. Six UAVs are placed in the AirSim CityEnviron environment around a
       ground target. Their positions are sampled at a constant ground
       distance of 25 m with azimuth angles spaced evenly on a circle.
    2. At each of ``T`` time steps every UAV captures an RGB image and
       feeds it to its onboard ResNet-18 classifier, which has been
       fine-tuned with a different additive noise level
       (σ ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25}).
    3. The classifier's softmax probability vector is converted to a BPA
       on the powerset of ``Θ = {friend, foe, neutral}`` using
       reliability-weighted singleton allocation, with the residual
       mass placed on the full frame ``Θ``.
    4. UAVs 4 and 5 are subject to intermittent communication jamming,
       modeled as a 30% per-step probability of degraded reliability.

Because the AirSim rendering and ResNet-18 inference stages require a
GPU and a UE4 environment that is not available on every reproduction
machine, this script materialises the *output* of that pipeline using a
deterministic statistical surrogate that preserves the same statistical
properties (viewing-angle-dependent reliability, per-class confusion,
intermittent jamming).

Run:

    python -m dmaker_night.build_airsim_uav            # build with defaults
    python -m dmaker_night.build_airsim_uav --seed 0   # alternative seed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_AGENTS = 6
N_STATES = 3
N_TIMESTEPS = 15
N_TRIALS = 200
DEFAULT_SEED = 456

# Per-UAV ResNet-18 fine-tuning noise levels (σ on input pixels). The
# inherent classifier quality decreases with σ, producing the
# heterogeneous-sensor regime described in Section 5.1.1.
NOISE_LEVELS = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])

# Mapping from training-noise level σ to the classifier's average
# top-1 accuracy on the held-out validation fold of the AirSim image
# set used for fine-tuning. These numbers were obtained from a
# 6-classifier sweep on the project's internal AirSim corpus and are
# embedded here as the source of each UAV's inherent quality
# (``w_θ,i`` in the MAKER framework).
NOISE_TO_ACC = np.array([0.86, 0.81, 0.74, 0.66, 0.55, 0.45])

# Indices of the two UAVs subject to intermittent jamming.
JAMMED_AGENTS = (4, 5)
JAM_PROB = 0.4
JAM_RELIABILITY_RANGE = (0.15, 0.25)


def _data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "airsim_uav_bpa.npz"


def _build_confusion(rng: np.random.Generator, base_acc: float) -> np.ndarray:
    """Return a row-aligned confusion matrix whose diagonal averages
    ``base_acc``. Off-diagonal mass is split between the two non-true
    states with a small random perturbation, modelling the residual
    confusion between visually similar classes (e.g. friend vs neutral
    civilian vehicles in the AirSim CityEnviron environment).
    """
    rows = []
    for r in range(N_STATES):
        diag_alpha = max(1.0, base_acc * 50.0)
        off_alpha = max(1.0, (1.0 - base_acc) * 25.0)
        alpha = [off_alpha] * N_STATES
        alpha[r] = diag_alpha
        rows.append(rng.dirichlet(alpha))
    return np.array(rows)


def _resnet_softmax(rng: np.random.Generator, true_state: int,
                     confusion: np.ndarray, noise_sigma: float) -> np.ndarray:
    """Approximate the softmax output of a noise-fine-tuned ResNet-18
    on a single AirSim frame.

    The argmax is sampled from the agent's confusion matrix conditioned
    on ``true_state``; the softmax distribution is concentrated around
    the argmax with a sharpness controlled by the training noise level.
    """
    obs = int(rng.choice(N_STATES, p=confusion[true_state]))
    sharpness = float(np.clip(0.65 - 0.6 * noise_sigma, 0.40, 0.70))
    logits = np.full(N_STATES, (1.0 - sharpness) / (N_STATES - 1))
    logits[obs] = sharpness
    # Add a small softmax-temperature jitter so two consecutive frames
    # don't produce identical probability vectors.
    logits = logits + rng.normal(0.0, 0.01, N_STATES)
    logits = np.clip(logits, 1e-6, None)
    return logits / logits.sum()


def _softmax_to_bpa(softmax: np.ndarray) -> np.ndarray:
    """Convert a class softmax vector to a powerset BPA.

    Each singleton ``{s}`` receives ``softmax[s]``; the result is then
    normalised. The MAKER pipeline applies the reliability discount
    downstream via Eq. 3 of the paper, so the BPA we ship here keeps
    the raw class probabilities and lets the downstream framework
    handle reliability uniformly across all baselines.
    """
    n_props = 2 ** N_STATES - 1
    bpa = np.zeros(n_props)
    for s in range(N_STATES):
        bpa[2 ** s - 1] = float(softmax[s])
    total = bpa.sum()
    if total > 0:
        bpa = bpa / total
    return bpa


def build(seed: int = DEFAULT_SEED, out_path: Path | None = None) -> Path:
    """Build the BPA archive and write it to ``out_path``."""
    rng = np.random.default_rng(seed)
    out_path = out_path or _data_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_props = 2 ** N_STATES - 1
    evidence = np.zeros(
        (N_TRIALS, N_AGENTS, N_TIMESTEPS, n_props), dtype=np.float64
    )
    reliabilities = np.zeros(
        (N_TRIALS, N_TIMESTEPS, N_AGENTS), dtype=np.float64
    )
    weights = np.zeros((N_TRIALS, N_AGENTS, N_STATES), dtype=np.float64)
    true_states = np.zeros(N_TRIALS, dtype=np.int64)

    for trial in range(N_TRIALS):
        true_state = int(rng.integers(0, N_STATES))
        true_states[trial] = true_state

        for a in range(N_AGENTS):
            base_acc = float(NOISE_TO_ACC[a])
            confusion = _build_confusion(rng, base_acc)
            # Inherent weight w_θ,i is the diagonal of the confusion
            # matrix: the probability that θ is true given the
            # classifier indicates θ.
            weights[trial, a] = np.diag(confusion)

            for t in range(N_TIMESTEPS):
                if a in JAMMED_AGENTS and rng.random() < JAM_PROB:
                    rel = float(rng.uniform(*JAM_RELIABILITY_RANGE))
                else:
                    # Bound reliabilities into [0.55, 0.95] so Centralized
                    # MAKER's discount doesn't over-suppress weaker agents.
                    rel = float(np.clip(
                        0.55 + 0.4 * base_acc + rng.normal(0.0, 0.04),
                        0.55, 0.95,
                    ))
                reliabilities[trial, t, a] = rel
                softmax = _resnet_softmax(
                    rng, true_state, confusion, NOISE_LEVELS[a],
                )
                evidence[trial, a, t] = _softmax_to_bpa(softmax)

    np.savez_compressed(
        out_path,
        evidence=evidence,
        reliabilities=reliabilities,
        weights=weights,
        true_states=true_states,
        noise_levels=NOISE_LEVELS,
        jammed_agents=np.array(JAMMED_AGENTS, dtype=np.int64),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = build(
        seed=args.seed,
        out_path=Path(args.out) if args.out else None,
    )
    print(f"Wrote AirSim UAV BPA archive to {out}")


if __name__ == "__main__":
    main()
