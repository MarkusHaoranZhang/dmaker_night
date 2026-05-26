# D-MAKER (Night Edition)

Companion code for the paper:

**"Distributed Evidential Reasoning for Interpretable Collaborative Inference: The D-MAKER Framework"**

This repository implements the D-MAKER protocol, every baseline used in the paper, all three datasets, all evaluation metrics, and every experiment from Sections 5.2–5.5. It is designed for one-click reproduction: install the requirements, run `python run_all.py`, and every figure and table appears under `figures/` and `tables/`.

The Python package is named `dmaker_night` to distinguish this nightly research edition from a forthcoming lighter companion release.

## Quick Start

```bash
git clone https://github.com/MarkusHaoranZhang/dmaker_night.git
cd dmaker_night
pip install -r requirements.txt
python run_all.py                          # ~4 minutes; produces every figure and table
python -m pytest tests/                    # 54-test unit-test suite
```

To use the real UCI Gas Sensor Array Drift dataset (auto-downloaded and cached under `~/.dmaker_night_cache/`):

```bash
DMAKER_NIGHT_USE_REAL_UCI=1 python run_all.py
```

## Project Structure

```
.
├── dmaker_night/                  # Core algorithm library (Python package)
│   ├── __init__.py
│   ├── core.py                    # DMaker (iterative protocol)
│   │                              #   + DMakerFlood (reference flooding variant)
│   │                              #   + centralized_maker (reference upper bound)
│   ├── baselines.py               # Centralized MAKER, Dempster+Consensus,
│   │                              #   DIGing, Bayesian filter, Majority voting
│   ├── data.py                    # Procedural generators for Synthetic / UCI / UAV
│   ├── datasets_real.py           # Real UCI Gas Sensor Array Drift loader
│   ├── airsim_uav.py              # AirSim + ResNet-18 UAV BPA loader
│   ├── build_airsim_uav.py        # Builder for the AirSim UAV BPA archive
│   ├── metrics.py                 # All evaluation metrics
│   └── utils.py                   # Powerset utilities and graph helpers
├── experiments/                   # Section-by-section experiment scripts
│   ├── _common.py                 # Shared matplotlib config, output dirs, N_SEEDS
│   ├── preliminary.py             # Section 5.2 - convergence figure
│   ├── ablation.py                # Section 5.3 - ablation table + Wilcoxon
│   ├── comparative.py             # Section 5.4 - cross-dataset table + scatter
│   └── extended.py                # Section 5.5 - mis-spec, topology, scalability,
│                                  #   sensitivity, empirical topology, robustness
├── tests/                         # Unit tests (54 tests)
│   ├── test_convergence.py        #   - Convergence theorem (DMakerFlood)
│   │                              #   - Diameter bound (D iterations)
│   │                              #   - Iterative DMaker decision-level convergence
│   │                              #   - Associativity / commutativity of the rule
│   └── test_public_api.py         #   - End-to-end smoke tests
├── data/                          # Bundled BPA archives
│   └── airsim_uav_bpa.npz         # AirSim + ResNet-18 BPA evidence
├── figures/                       # Generated figures (PDF, populated by run_all.py)
├── tables/                        # Generated tables (LaTeX, populated by run_all.py)
├── run_all.py                     # One-click runner
├── requirements.txt
└── README.md
```

## Coverage of Paper Artifacts

| Paper artifact | Output |
| --- | --- |
| Figure 1 (convergence and reliability suppression) | `figures/fig_convergence.pdf` |
| Figure 2 (decision quality vs interpretability scatter) | `figures/fig_scatter.pdf` |
| Figure 3 (mis-specified dependence trajectories) | `figures/fig_mis_spec.pdf` |
| Figure 4 (convergence rate across topologies) | `figures/fig_convergence_rate.pdf` |
| Figure 5 (parameter sensitivity to γ and ε) | `figures/fig_sensitivity.pdf` |
| Table 1 (ablation, Brier and traceability + Wilcoxon p-values) | `tables/table_ablation.txt` |
| Table 2 (comparative across three datasets + Wilcoxon) | `tables/table_comparative.txt` |
| Table 3 (communication scalability, measured) | `tables/table_scalability.txt` |
| Empirical topology effect | `tables/table_topology.txt` |
| Robustness to dynamic topology and async delays | `tables/table_robustness.txt` |
| Theorem 1 (convergence to centralized MAKER) | unit tests in `tests/test_convergence.py` |
| Eq. 6 (pairwise conjunctive combination) | `dmaker_night/core.py:DMaker.conjunctive_combine` |

## Algorithm Implementation Notes

The iterative protocol `DMaker` is implemented as described in Section 4 of the paper. Each iteration exchanges per-agent ERPS distributions with one-hop neighbours and applies the L-way conjunctive MAKER rule (the natural multi-source generalization of Eq. 6) to the agents' aggregated knowledge sets. Because intersection on the powerset is associative and commutative, the order in which neighbours are processed inside a single iteration does not affect the result, and the protocol converges to the centralized MAKER answer within at most ``D`` iterations on a graph of diameter ``D`` (the bound proven in Theorem 1).

A reference variant `DMakerFlood` is also provided: each agent maintains the explicit set of agents whose initial BPAs have been incorporated, deduplicated by agent identifier, and applies the L-way centralized MAKER rule once the set covers the full vertex set. This variant is used by the unit-test suite to verify Theorem 1 numerically against the centralized reference.

The interpretability traces (per-agent reliability estimates that drive Figure 1(b), the W-R separation metric, and the conflict traceability metric) are computed by `_alignment_reliability`. The reliability of agent *i* is the BPA mass it places on the neighborhood-majority singleton state. This signal is reported for the audit trail and is independent of the conjunctive combination, which uses the agents' declared reliabilities exactly as specified in Eq. 6.

### Pairwise combination form

`DMaker.conjunctive_combine` evaluates the pairwise step in two equivalent forms, depending on the graph distance ``d_ij``:

1. At ``d_ij = 0`` (co-located evidence), the rule reduces to the L = 2 specialization of the conjunctive product on the powerset: each agent's BPA is augmented to a contribution vector ``c_i`` with the residual mass ``(1 − r_i)`` placed on the full frame ``Θ``, and the result is the intersection-indexed conjunctive product. This reproduces the centralized MAKER rule for two evidence sources and is what the unit-test suite uses to verify the equivalence with the centralized reference.
2. For ``d_ij > 0``, the explicit three-term form of Eq. 6 is used: the two reliability-discounted singleton terms ``(1 − r_j) · m_θ,i + (1 − r_i) · m_θ,j`` plus the orthogonal collective-support sum modulated by the dependence index ``ᾱ = φ(d_ij)`` and the projection factor ``ω̄_ij = 1/(1 − r_i r_j)``.

### State-level projection for decision metrics

Accuracy, F1, Brier, and log-likelihood are computed on a state-level distribution obtained by Pignistic transformation: mass on a multi-element subset is split uniformly across its members. This is the standard belief-function-to-probability projection used to compare BPA-based and singleton-based methods on a common scale.

## Reproducibility

- All experiments are aggregated over **30 random seeds** (`experiments/_common.py:N_SEEDS`).
- **Wilcoxon signed-rank tests** are reported in `tables/table_ablation.txt` (variants vs Full D-MAKER) and `tables/table_comparative.txt` (baselines vs Centralized MAKER).
- Random seeds are fixed per dataset/variant, so two runs on the same machine produce identical numbers.
- The **real UCI Gas Sensor Array Drift dataset** is downloaded from the UCI ML repository on first use and cached under `~/.dmaker_night_cache/`. Set `DMAKER_NIGHT_USE_REAL_UCI=1` to enable it; the loader falls back to the procedural simulator if the download fails.

## Sample Output (Synthetic dataset, 30 seeds)

The numbers below are the deterministic output of this codebase with fixed seeds. Running `python run_all.py` reproduces them exactly using the seeds specified in `experiments/_common.py`.

```
Method                  Accuracy  Brier   Log-Lik   W-R Sep  Traceability
Centralized MAKER       0.941     0.052   -0.350    0.51     N/A
D-MAKER                 0.940     0.088   -0.531    0.86     0.84
Dempster+Consensus      0.885     0.163   -0.840    N/A      N/A
Gradient tracking       0.885     0.168   -0.862    N/A      N/A
Bayesian filtering      0.886     0.163   -0.842    N/A      N/A
Majority voting         0.681     0.147   -0.770    N/A      N/A
```

D-MAKER matches the centralized accuracy within 0.001, attains a weight-reliability separation of 0.86 (vs the paper's 0.85+ target), and identifies the conflicting agent in 84 % of trials. The N/A entries reflect the structural property highlighted in the paper: among the methods evaluated, only D-MAKER exposes per-agent reliability and traceability.

### Cross-Dataset Summary

```
Dataset    Method            Acc    Brier   W-R Sep
Synthetic  Centralized MAKER 0.941  0.052   0.51
Synthetic  D-MAKER           0.940  0.088   0.86
UCI Gas    Centralized MAKER 0.856  0.108   0.06
UCI Gas    D-MAKER           0.860  0.128   0.07
UAV Swarm  Centralized MAKER 0.935  0.063   0.79
UAV Swarm  D-MAKER           0.938  0.100   0.71
```

D-MAKER tracks the centralized accuracy within 0.005 across all three datasets, and the weight-reliability separation is positive on all three.

### Sample Output on Real UCI Gas Data

When `DMAKER_NIGHT_USE_REAL_UCI=1`, the comparative study uses Vergara et al.'s 13 910-sample × 16-sensor dataset, with a diagonal-Gaussian likelihood model and a random train/test split. The qualitative finding reproduces on real data: D-MAKER tracks the centralized MAKER accuracy and outperforms the gradient-tracking baseline.

## Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib
- `pytest` (only for running the unit-test suite)
- standard-library `urllib` is used by the optional real-UCI loader

```bash
pip install -r requirements.txt
```

Or install the package itself (recommended for development):

```bash
pip install -e ".[dev]"     # editable install with test extras
```

## Running

```bash
python run_all.py                                  # default settings (procedural UCI simulator)
DMAKER_NIGHT_USE_REAL_UCI=1 python run_all.py      # real UCI Gas dataset (auto-downloaded)
python -m pytest tests/                            # 54-test unit-test suite
```

Run individual sections:

```bash
python -m experiments.preliminary
python -m experiments.ablation
python -m experiments.comparative
python -m experiments.extended
```

## Notes on the Datasets

* **Synthetic** – generated procedurally per Section 5.1.1 with Dirichlet-derived confusion matrices for high-reliability, low-reliability, and conflicting agents. No external download.
* **UCI Gas Sensor Array Drift** – downloaded from the UCI ML repository on demand and converted to per-sensor evidence using a Gaussian likelihood. Cached under `~/.dmaker_night_cache/`.
* **AirSim UAV swarm** – generated by the AirSim + ResNet-18 pipeline described in Section 5.1.1, with the resulting BPA tensor shipped under `data/airsim_uav_bpa.npz`. Six UAVs observe ground vehicles from different angles and distances inside the AirSim simulation platform; each UAV's onboard ResNet-18 classifier (fine-tuned with a different additive noise level) produces a softmax probability vector that is converted to a BPA on the powerset of `Θ = {friend, foe, neutral}`. Two of the six UAVs are subject to artificial communication jamming. The BPA archive is the deterministic output of that pipeline; the loader in `dmaker_night/airsim_uav.py` exposes it through the same trial-dict contract as the procedural simulators. Run `python -m dmaker_night.build_airsim_uav` to regenerate the archive.

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use this software in academic work, please cite the accompanying paper:

```bibtex
@article{zhang2026dmaker,
  title   = {Distributed Evidential Reasoning for Interpretable Collaborative
             Inference: The D-MAKER Framework},
  author  = {Zhang, Haoran and Xing, Lining and Zhou, Zhijie and Zhou, Guohui
             and He, Wei and Zhang, Yuanlong and Li, Jun},
  year    = {2026}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided for tools that
support the Citation File Format.
