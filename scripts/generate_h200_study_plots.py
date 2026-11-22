import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#07111f",
    "axes.facecolor": "#0d1b2e",
    "axes.edgecolor": "#24415f",
    "axes.labelcolor": "#e8f0f7",
    "xtick.color": "#9eb2c7",
    "ytick.color": "#9eb2c7",
    "text.color": "#e8f0f7",
    "grid.color": "#1c334d",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
})

# ---------------------------------------------------------
# 1. B08: Concurrency Pareto Curve (Throughput vs TTFT)
# ---------------------------------------------------------
concurrency = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256]
throughput = [243.62, 498.23, 966.68, 1863.00, 3423.20, 5297.67, 8418.35, 10382.34, 10266.93, 11449.47]
ttft_p50 = [49.92, 21.36, 25.49, 28.33, 29.21, 32.57, 42.87, 122.45, 535.58, 1013.83]

x = np.arange(len(concurrency))
fig, ax1 = plt.subplots(figsize=(10, 5.2), dpi=300)

ax1.axvspan(4, 6, color="#34d399", alpha=0.12, label="Interactive Sweet Spot (c=16–64)")
l1 = ax1.plot(x, throughput, marker="o", linewidth=2.8, color="#34d399", label="Output Throughput (tok/s)")
ax1.set_xlabel("Concurrent Requests (max-concurrency)", fontweight="bold", fontsize=11)
ax1.set_ylabel("Output Throughput (tokens/sec)", color="#34d399", fontweight="bold", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels([str(c) for c in concurrency])
ax1.grid(True)

ax2 = ax1.twinx()
l2 = ax2.plot(x, ttft_p50, marker="s", linewidth=2.5, linestyle="-.", color="#fb7185", label="Median TTFT (ms)")
ax2.set_ylabel("Median TTFT (ms)", color="#fb7185", fontweight="bold", fontsize=11)
ax2.spines["top"].set_visible(False)
ax1.spines["top"].set_visible(False)

lines = l1 + l2
labels = [l.get_label() for l in lines] + ["Interactive Sweet Spot (c=16–64)"]
ax1.legend(lines + [plt.Rectangle((0, 0), 1, 1, fc="#34d399", alpha=0.2)], labels, loc="upper left", frameon=True, facecolor="#0d1b2e", edgecolor="#24415f")

plt.title("NVIDIA H200 NVL (SM 9.0): Concurrency Scaling vs. TTFT Latency (B08)", fontweight="bold", fontsize=13, pad=14)
plt.tight_layout()
plt.savefig("assets/h200_concurrency_pareto.png")
plt.close()

# ---------------------------------------------------------
# 2. B07: FP8 vs BF16 Precision Ablation (Weights x KV)
# ---------------------------------------------------------
configs = ["BF16 / BF16\n(Baseline)", "BF16 / FP8\n(KV Quant)", "FP8 / BF16\n(Weight Quant)", "FP8 / FP8\n(Full FP8)"]
c32_vals = [4544.88, 4580.98, 5159.55, 5267.88]
c128_vals = [9201.76, 9780.49, 10190.54, 10436.95]

x2 = np.arange(len(configs))
width = 0.36

fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=300)
b1 = ax.bar(x2 - width / 2, c32_vals, width, label="Concurrency = 32 (+15.9% FP8/FP8)", color="#22d3ee")
b2 = ax.bar(x2 + width / 2, c128_vals, width, label="Concurrency = 128 (+13.4% FP8/FP8)", color="#a78bfa")

for bar in b1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 180, f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#22d3ee")
for bar in b2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 180, f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#a78bfa")

ax.set_ylabel("Output Throughput (tokens/sec)", fontweight="bold", fontsize=11)
ax.set_xticks(x2)
ax.set_xticklabels(configs, fontweight="bold")
ax.set_ylim(0, 11800)
ax.grid(True, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=True, facecolor="#0d1b2e", edgecolor="#24415f")

plt.title("H200 NVL Precision Ablation: Weights vs. KV-Cache Quantization (B07)", fontweight="bold", fontsize=13, pad=14)
plt.tight_layout()
plt.savefig("assets/h200_fp8_precision_ablation.png")
plt.close()

# ---------------------------------------------------------
# 3. B09 & B10: Prefix Caching & Chunked Prefill Trade-off
# ---------------------------------------------------------
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

# Left: B09 Prefix Caching
patterns = ["8 Prefix\nGroups", "Identical\nPrefix", "Unique\nPrefixes"]
latencies = [206.58, 234.56, 1664.39]
colors_l = ["#34d399", "#22d3ee", "#fb7185"]
bars_l = ax_l.bar(patterns, latencies, color=colors_l, width=0.52)
for bar in bars_l:
    ax_l.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 35, f"{bar.get_height():.1f} ms", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax_l.set_ylabel("Mean Request Latency (ms)", fontweight="bold")
ax_l.set_ylim(0, 1900)
ax_l.grid(True, axis="y")
ax_l.spines["top"].set_visible(False)
ax_l.spines["right"].set_visible(False)
ax_l.set_title("B09: Prefix Caching Speedup (7.1x)", fontweight="bold", fontsize=11.5)

# Right: B10 Chunked Prefill
batched_tokens = ["2,048", "4,096", "8,192", "16,384", "32,768", "65,536"]
ttft_b10 = [268, 442, 710, 1120, 1530, 1844]
out_b10 = [1403, 1480, 1550, 1610, 1284, 1863]

xr = np.arange(len(batched_tokens))
ax_r.plot(xr, ttft_b10, marker="o", linewidth=2.5, color="#fbbf24", label="Median TTFT (ms)")
ax_r.set_xticks(xr)
ax_r.set_xticklabels(batched_tokens, rotation=20)
ax_r.set_xlabel("max-num-batched-tokens", fontweight="bold")
ax_r.set_ylabel("Median TTFT (ms)", color="#fbbf24", fontweight="bold")
ax_r.grid(True)
ax_r.spines["top"].set_visible(False)

ax_r2 = ax_r.twinx()
ax_r2.plot(xr, out_b10, marker="^", linewidth=2.2, linestyle="--", color="#22d3ee", label="Output (tok/s, c=32)")
ax_r2.set_ylabel("Output Throughput (tok/s)", color="#22d3ee", fontweight="bold")
ax_r2.spines["top"].set_visible(False)
ax_r.set_title("B10: Chunked Prefill SLO Trade-off (c=32)", fontweight="bold", fontsize=11.5)

plt.tight_layout()
plt.savefig("assets/h200_prefix_and_chunked_prefill.png")
plt.close()
print("[OK] Generated 3 H200 characterization plots in assets/")