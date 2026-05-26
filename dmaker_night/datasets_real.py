"""Loaders for real datasets used in the D-MAKER experiments.

Currently supports:

* UCI Gas Sensor Array Drift Dataset (Vergara et al., 2012). The data
  consists of 13,910 samples × 128 features × 6 gas classes collected
  from 16 metal-oxide sensors over 36 months. We reduce the dataset to
  per-sensor likelihood evidence to fit the D-MAKER framework.

The loader downloads the raw archive from the UCI ML repository on
first use, caches it under ``~/.dmaker_night_cache``, and converts every
sample to a ``BPA × n_states`` evidence vector.

If the download fails (no internet, mirror down, etc.) the loader
raises a ``RuntimeError`` so the calling experiment can fall back to
the procedural simulator in ``dmaker_night/data.py``.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np

# Public mirror with stable URLs for the UCI Gas Sensor Array Drift
# dataset. The first entry is the original UCI archive; subsequent
# entries are widely-mirrored copies that we fall back to if the
# primary mirror is unreachable.
_UCI_GAS_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip",
    "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
]

_CACHE_DIR = Path(os.environ.get("DMAKER_NIGHT_CACHE", Path.home() / ".dmaker_night_cache"))


def _download(url: str, dest: Path, timeout: int = 60) -> None:
    """Download ``url`` to ``dest``. Uses the standard library only."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; D-MAKER experimental code; "
                "academic reproduction)"
            ),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dest.write_bytes(data)


def _load_uci_gas_raw(force_download: bool = False) -> np.ndarray:
    """Return the raw UCI Gas Sensor Array Drift data as a NumPy array.

    Each row is one sample and the columns are
    ``[class, batch, feature_1, ..., feature_128]``. ``class`` is in
    1..6 and ``batch`` is in 1..10.
    """
    cache_root = _CACHE_DIR / "uci_gas"
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_npy = cache_root / "uci_gas.npy"
    if cached_npy.exists() and not force_download:
        return np.load(cached_npy)

    archive_path = cache_root / "Dataset.zip"
    last_err: Exception | None = None
    downloaded = False
    for url in _UCI_GAS_URLS:
        try:
            _download(url, archive_path)
            downloaded = True
            break
        except Exception as exc:  # pragma: no cover - network dependent
            last_err = exc
    if not downloaded:
        raise RuntimeError(
            f"Failed to download UCI Gas Sensor Array Drift dataset from any "
            f"mirror; last error: {last_err}"
        )

    rows = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        # The official archive contains 10 batch files: batch1.dat, ...
        for member in zf.namelist():
            if not member.lower().endswith(".dat"):
                continue
            try:
                batch_id = int(
                    member.lower().split("batch")[-1].split(".dat")[0]
                )
            except ValueError:
                continue
            with zf.open(member) as fh:
                for line in fh:
                    text = line.decode("ascii", errors="ignore").strip()
                    if not text:
                        continue
                    # Format: "class;cls_id feat_id:value feat_id:value ..."
                    parts = text.replace(";", " ").split()
                    if len(parts) < 2:
                        # Malformed line — skip rather than aborting the parse.
                        continue
                    try:
                        cls = int(parts[0])
                    except ValueError:
                        # Header / comment / non-numeric line — skip.
                        continue
                    feats = np.zeros(128, dtype=float)
                    for token in parts[1:]:
                        if ":" not in token:
                            continue
                        idx, val = token.split(":")
                        try:
                            feats[int(idx) - 1] = float(val)
                        except (ValueError, IndexError):
                            continue
                    rows.append([cls, batch_id, *feats.tolist()])
    if not rows:
        raise RuntimeError("UCI Gas archive parsed but no rows extracted.")
    arr = np.asarray(rows, dtype=float)
    np.save(cached_npy, arr)
    return arr


def load_uci_gas_evidence(
    n_trials: int = 30,
    n_sensors: int = 16,
    n_states: int = 6,
    seed: int = 123,
    force_download: bool = False,
):
    """Convert the UCI Gas Sensor Array Drift dataset into D-MAKER trials.

    Returns a list of trial dicts compatible with :class:`SyntheticDataset`.

    For each test sample we form one trial:
    - ``true_state`` is the true gas class (0..n_states-1)
    - ``evidence[a]`` is a single-step BPA per sensor ``a``, derived from
      the Gaussian likelihood of the sensor's 8 features under each of the
      ``n_states`` Gaussian class models fitted on the training fold
    - ``reliabilities[a]`` is the per-sensor classification accuracy on
      the training fold (a stable, observable proxy for sensor quality)
    - ``weights[a]`` is the per-class accuracy of the sensor on the
      training fold (i.e. ``w_θ,i = P(θ correct | sensor a indicates θ)``)
    """
    raw = _load_uci_gas_raw(force_download=force_download)
    classes = raw[:, 0].astype(int) - 1  # zero-indexed
    features = raw[:, 2:]  # drop class and batch columns
    feats_per_sensor = features.shape[1] // n_sensors

    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]
    perm = rng.permutation(n_samples)
    train_idx = perm[: n_samples // 2]
    test_idx = perm[n_samples // 2: n_samples // 2 + n_trials]

    # Fit per-sensor per-class Gaussian likelihood on the training fold.
    sensor_means = np.zeros((n_sensors, n_states, feats_per_sensor))
    sensor_vars = np.ones((n_sensors, n_states, feats_per_sensor))
    for a in range(n_sensors):
        cols = slice(a * feats_per_sensor, (a + 1) * feats_per_sensor)
        for c in range(n_states):
            mask = classes[train_idx] == c
            if mask.sum() < 2:
                continue
            sub = features[train_idx][mask, cols]
            sensor_means[a, c] = sub.mean(axis=0)
            sensor_vars[a, c] = sub.var(axis=0) + 1e-3

    # Compute per-sensor reliability and weights from training accuracy.
    reliabilities = np.zeros(n_sensors)
    weights = np.zeros((n_sensors, n_states))
    for a in range(n_sensors):
        cols = slice(a * feats_per_sensor, (a + 1) * feats_per_sensor)
        log_probs = _gaussian_log_likelihood(
            features[train_idx][:, cols], sensor_means[a], sensor_vars[a]
        )
        preds = np.argmax(log_probs, axis=1)
        truth = classes[train_idx]
        reliabilities[a] = float((preds == truth).mean())
        for c in range(n_states):
            mask = preds == c
            if mask.sum() == 0:
                weights[a, c] = 0.5
            else:
                weights[a, c] = float((truth[mask] == c).mean())

    trials = []
    n_props = 2 ** n_states - 1
    for sample_idx in test_idx:
        true_state = int(classes[sample_idx])
        evidence = []
        for a in range(n_sensors):
            cols = slice(a * feats_per_sensor, (a + 1) * feats_per_sensor)
            x = features[sample_idx, cols]
            log_p = _gaussian_log_likelihood(
                x[None, :], sensor_means[a], sensor_vars[a]
            )[0]
            log_p = log_p - log_p.max()
            posterior = np.exp(log_p)
            posterior = posterior / posterior.sum()
            bpa = np.zeros(n_props)
            for c in range(n_states):
                bpa[2 ** c - 1] = posterior[c] * reliabilities[a]
            bpa[-1] += 1.0 - reliabilities[a]
            bpa = bpa / bpa.sum()
            # Single time step (the dataset gives one reading per sensor)
            evidence.append([bpa])
        trials.append({
            "true_state": true_state,
            "evidence": evidence,
            "reliabilities": reliabilities.copy(),
            "weights": [weights[a].copy() for a in range(n_sensors)],
        })
    return trials, reliabilities, weights


def _gaussian_log_likelihood(X, means, variances):
    """Vectorized diagonal-Gaussian log-likelihood for X ∈ R^{n_samples × d}.

    Returns an ``n_samples × n_classes`` log-likelihood array.
    """
    # X shape: (n, d)
    # means / variances shape: (C, d)
    n, d = X.shape
    C = means.shape[0]
    diff = X[:, None, :] - means[None, :, :]
    log_lik = -0.5 * np.sum(
        diff ** 2 / variances[None, :, :] + np.log(2 * np.pi * variances[None, :, :]),
        axis=-1,
    )
    assert log_lik.shape == (n, C)
    return log_lik
