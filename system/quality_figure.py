import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
SAGE = {
    "LLaMA-3.2-1B": {"x": [5.05, 10.06, 15.08, 40.11], "y": [0.3634, 0.4883, 0.5624, 0.6732]},
    "Qwen3-8B":     {"x": [5.05, 10.08, 15.07, 40.12], "y": [0.4629, 0.5493, 0.5953, 0.6808]},
    "Qwen3-14B":    {"x": [5.05, 10.06, 15.09, 40.11], "y": [0.4770, 0.5859, 0.6150, 0.7042]},
}
RAG_BASELINES = {
    "Qwen3-Embedding-8B":  {"x": [12.02, 20.01, 28.04, 40.10], "y": [0.4920, 0.5737, 0.5822, 0.6075]},
    "UAE-Large-V1":        {"x": [12.07, 20.11, 28.15, 40.24], "y": [0.4413, 0.5108, 0.5653, 0.6094]},
    "Octen-Embedding-4B":  {"x": [12.07, 20.12, 28.14, 40.19], "y": [0.4930, 0.5549, 0.5991, 0.6235]},
}
AT_COLORS   = ["#1a5276", "#2e86c1", "#7fb3d3"]
RAG_COLORS  = ["#922b21", "#e03c31", "#f1948a"]
AT_MARKERS  = ["o", "s", "^"]
RAG_MARKERS = ["D", "v", "P"]
LW, MS = 2.0, 6
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(True, alpha=0.22, linestyle="--", color="gray")
ax.set_axisbelow(True)
at_handles = []
for (name, d), color, marker in zip(SAGE.items(), AT_COLORS, AT_MARKERS):
    ax.plot(d["x"], d["y"], color=color, marker=marker, linestyle="-",
            linewidth=LW, markersize=MS, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    at_handles.append(mlines.Line2D([], [], color=color, marker=marker, linestyle="-",
            linewidth=LW, markersize=MS+1, markeredgecolor="white", markeredgewidth=0.8, label=name))
rag_handles = []
for (name, d), color, marker in zip(RAG_BASELINES.items(), RAG_COLORS, RAG_MARKERS):
    ax.plot(d["x"], d["y"], color=color, marker=marker, linestyle="--",
            linewidth=LW, markersize=MS, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    rag_handles.append(mlines.Line2D([], [], color=color, marker=marker, linestyle="--",
            linewidth=LW, markersize=MS+1, markeredgecolor="white", markeredgewidth=0.8, label=name))

# Section title handles (invisible lines used as bold headers)
_at_title  = mlines.Line2D([], [], linestyle="None", markersize=0, color="none", label="SAGE")
_rag_title = mlines.Line2D([], [], linestyle="None", markersize=0, color="none", label="RAG Baselines")

all_handles = [_at_title] + at_handles + [_rag_title] + rag_handles
all_labels  = (["SAGE"] + [h.get_label() for h in at_handles] +
               ["RAG Baselines"] + [h.get_label() for h in rag_handles])

legend = ax.legend(
    handles=all_handles, labels=all_labels,
    loc="lower right",
    ncol=1, fontsize=9.5, frameon=True, edgecolor="#cccccc",
    fancybox=False, handlelength=2.2, handleheight=1.5,
    borderpad=0.6, labelspacing=0.3, columnspacing=1.0,
)

# Bold the section title entries
texts = legend.get_texts()
for idx in (0, 4):
    texts[idx].set_fontweight("bold")
    texts[idx].set_fontsize(9.5)

ax.set_xlim(0, 44)
ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
ax.tick_params(labelsize=13)
ax.set_ylim(0.30, 0.74)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.set_xlabel("Avg. Token Usage (%)", fontsize=16, labelpad=8)
ax.set_ylabel("Avg. Accuracy", fontsize=16, labelpad=8)
plt.tight_layout()
plt.savefig("figures/quality_hard_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("figures/quality_hard_comparison.pdf", bbox_inches="tight", facecolor="white")
print("Saved.")