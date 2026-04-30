import json
import numpy as np
from scipy.io import loadmat
from task_and_baseline import baseline, build_task_helpers

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):
    score_filter = helpers["score_filter"]

    # 1. TX-часть через официальную модель
    tx_part = helpers["fit_tx_prediction"](rx)
    residual = rx - tx_part

    # 2. Rank-1 остатка — точно как в rank1_from_band_matrix
    band = np.column_stack([score_filter(residual[:, ch]) for ch in range(4)])
    cov = band.conj().T @ band / band.shape[0]
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    shared = band @ v
    denom = np.vdot(shared, shared) + 1e-30
    weights = np.array([np.vdot(shared, band[:, ch]) / denom for ch in range(4)])
    rank1 = np.outer(shared, weights)

    return rx - tx_part - rank1


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {"per_channel_db": baseline_reds, "average_db": baseline_avg},
    "yours":    {"per_channel_db": yours_reds,    "average_db": yours_avg},
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)