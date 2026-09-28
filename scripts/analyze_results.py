"""Base-vs-SFT statistics from results/accuracy.csv (n = 1,000 per condition): delta, 95% CI,
two-proportion z-test with Holm correction, transfer gap G = delta(L) - delta(EN), retention ratio rho = A(L)/A(EN)."""
import csv
from math import sqrt, erf
N = 1000
rows = list(csv.DictReader(open("results/accuracy.csv")))
for r in rows: r["b"], r["s"] = float(r["base_acc"]), float(r["sft_acc"])
en = {r["model"]: r for r in rows if r["language"] == "EN"}
pval = lambda z: 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
for r in rows:
    b, s = r["b"] / 100, r["s"] / 100
    se = sqrt(b * (1 - b) / N + s * (1 - s) / N)
    r["delta"] = r["s"] - r["b"]; r["p"] = pval((s - b) / se)
    r["ci_lo"], r["ci_hi"] = r["delta"] - 196 * se, r["delta"] + 196 * se
    e = en[r["model"]]
    r["gap"] = r["delta"] - (e["s"] - e["b"])
    r["rho_base"], r["rho_sft"] = r["b"] / e["b"], r["s"] / e["s"]
running = 0.0
for k, r in enumerate(sorted(rows, key=lambda r: r["p"])):
    running = max(running, min(1.0, (len(rows) - k) * r["p"])); r["p_holm"] = running
cols = ["model", "language", "base_acc", "sft_acc", "delta", "ci_lo", "ci_hi", "p", "p_holm", "gap", "rho_base", "rho_sft"]
with open("results/statistics.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in rows:
        w.writerow([r[c] if c in cols[:4] else (f"{r[c]:.3g}" if c.startswith("p") else f"{r[c]:.2f}") for c in cols])
print(open("results/statistics.csv").read())
