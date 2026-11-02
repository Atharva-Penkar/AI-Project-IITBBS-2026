"""Regenerate knowledge_graph.png from knowledge_graph.json (needs matplotlib).

    python -m module1_knowledge_graph.draw_graph
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_knowledge_graph.knowledge_graph import nodes, prereqs

layer = {}


def level(c):
    if c not in layer:
        layer[c] = 0 if not prereqs[c] else 1 + max(level(p) for p in prereqs[c])
    return layer[c]


for c in nodes:
    level(c)
layers = {}
for c in nodes:
    layers.setdefault(layer[c], []).append(c)

order = {}
for L in sorted(layers):
    if L == 0:
        layers[L].sort()
    else:
        def bary(c):
            ps = [order[p] for p in prereqs[c] if p in order]
            return sum(ps) / len(ps) if ps else 0
        layers[L].sort(key=lambda c: (bary(c), c))
    for i, c in enumerate(layers[L]):
        order[c] = i

pos = {}
for L, members in layers.items():
    for i, c in enumerate(members):
        pos[c] = (L * 3.6, -(i - (len(members) - 1) / 2) * 1.8)

fig, ax = plt.subplots(figsize=(16, 9))
for c in nodes:
    for p in prereqs[c]:
        (x1, y1), (x2, y2) = pos[p], pos[c]
        ax.annotate("", xy=(x2 - 1.0, y2), xytext=(x1 + 1.0, y1), zorder=1,
                    arrowprops=dict(arrowstyle="->", color="gray"))
for c, (x, y) in pos.items():
    ax.text(x, y, nodes[c]["name"], ha="center", va="center", fontsize=9, zorder=3,
            bbox=dict(boxstyle="round,pad=0.4", fc="#e8f0fe", ec="#1a73e8"))
xs = [x for x, _ in pos.values()]
ys = [y for _, y in pos.values()]
ax.set_xlim(min(xs) - 2, max(xs) + 2)
ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
ax.axis("off")
ax.set_title("AI Search Knowledge Graph (arrow = prerequisite of)")
out_path = Path(__file__).resolve().parent / "knowledge_graph.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"wrote {out_path}")
