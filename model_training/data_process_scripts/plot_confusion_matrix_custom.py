"""Plot confusion matrix (sklearn layout: rows=true, cols=predicted). Fake=class 0, Real=class 1."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from pathlib import Path

# Rows: true Fake, true Real. Cols: pred Fake, pred Real.
cm = np.array([[75, 0], [1, 74]], dtype=int)

out = Path(__file__).resolve().parent / "confusion_matrix_custom.png"
fig, ax = plt.subplots(figsize=(6.2, 5.2))
cmap = plt.cm.Greens
norm = mcolors.Normalize(vmin=cm.min(), vmax=cm.max())
im = ax.imshow(cm, cmap=cmap, norm=norm)
cbar = fig.colorbar(im, ax=ax)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Fake", "Real"])
ax.set_yticklabels(["Fake", "Real"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix (Test Set)")

for i in range(2):
    for j in range(2):
        val = int(cm[i, j])
        rgba = cmap(norm(val))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        txt_color = "white" if lum < 0.55 else "#2d2d2d"
        ax.text(
            j,
            i,
            val,
            ha="center",
            va="center",
            color=txt_color,
            fontsize=10,
            fontweight="medium",
        )

ax.tick_params(axis="both", which="major", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
