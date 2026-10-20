"""Figures. Colors follow the dataviz reference palette: one-hue blue sequential ramp
for accuracy (magnitude), text in ink tokens, non-significance as texture + label
(never color alone)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def _luminance(rgb) -> float:
    c = np.array(rgb)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _draw_cells(ax, acc: np.ndarray, sig: np.ndarray, norm, fontsize: float = 8, outline_diag: bool = False):
    """Value labels on each cell; non-significant cells get a hatch + 'n.s.' (not color alone)."""
    R, C = acc.shape
    for i in range(R):
        for j in range(C):
            v = acc[i, j]
            if np.isnan(v):
                continue
            fill = SEQ(norm(v))[:3]
            ink = SURFACE if _luminance(fill) < 0.28 else INK
            if not sig[i, j]:
                ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fill=False, hatch="////",
                                           edgecolor=to_rgb(MUTED), linewidth=0))
            label = f"{v:.1f}" if sig[i, j] else f"{v:.1f}\nn.s."
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=fontsize, color=ink,
                    linespacing=0.9, fontweight="bold" if sig[i, j] else "normal")
            if outline_diag and i == j:
                ax.add_patch(plt.Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, fill=False, edgecolor=INK,
                                           linewidth=1.2))


def accuracy_heatmap(acc: np.ndarray, significant: np.ndarray, row_labels: list[str], col_labels: list[str],
                     path: Path, title: str, chance: float = 100 / 7, gap_after: int | None = None,
                     gap_label: str = "", note: str = "", xlabel: str = "Behavior",
                     ylabel: str = "Participant – array", ci: np.ndarray | None = None) -> None:
    """acc in percent (R x C); NaN cells are drawn empty. Optional gap row after `gap_after` rows."""
    acc = np.asarray(acc, float)
    sig = np.asarray(significant, bool)
    rows = list(row_labels)
    if gap_after is not None and gap_after < len(rows):
        acc = np.insert(acc, gap_after, np.nan, axis=0)
        sig = np.insert(sig, gap_after, True, axis=0)
        if ci is not None:
            ci = np.insert(np.asarray(ci, float), gap_after, np.nan, axis=0)
        rows.insert(gap_after, "")
    R, C = acc.shape
    fig, ax = plt.subplots(figsize=(1.15 * C + 2.6, 0.42 * R + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    norm = matplotlib.colors.Normalize(vmin=chance, vmax=100)
    mesh = ax.pcolormesh(np.ma.masked_invalid(acc), cmap=SEQ, norm=norm, edgecolors=SURFACE, linewidth=2)
    _draw_cells(ax, acc, sig, norm)
    ax.set_xticks(np.arange(C) + 0.5, col_labels, rotation=30, ha="right", fontsize=8.5, color=INK_2)
    ax.set_yticks(np.arange(R) + 0.5, rows, fontsize=8.5, color=INK_2)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if gap_after is not None and gap_label:
        ax.text(-0.1, gap_after + 0.5, gap_label, ha="right", va="center", fontsize=7.5, color=MUTED,
                style="italic", transform=ax.transData)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    cb = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02)
    cb.set_ticks([chance, 25, 50, 75, 100], labels=[f"chance\n{chance:.1f}", "25", "50", "75", "100"])
    cb.ax.tick_params(labelsize=7.5, colors=INK_2, length=0)
    cb.outline.set_visible(False)
    cb.set_label("accuracy (%)", color=INK_2, fontsize=8.5)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def matrix_grid(panels: list[dict], row_labels: list[str], col_labels: list[str], path: Path, title: str,
                ncols: int, chance: float = 100 / 7, note: str = "", xlabel: str = "test behavior",
                ylabel: str = "train behavior") -> None:
    """Small multiples of train x test accuracy matrices sharing one color scale.

    panels: [{"acc": (R, C) %, "sig": (R, C) bool, "title": str, "labels": optional per-panel
    labels for both axes}]; diagonal cells are outlined.
    """
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    R, C = np.asarray(panels[0]["acc"]).shape
    cell = 0.52 if C > 4 else 0.75
    own_labels = any("labels" in p for p in panels)  # every panel needs room for its own tick labels
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * (C * cell + 1.9) + 0.8, nrows * (R * cell + 1.5) + 0.6),
                             facecolor=SURFACE, squeeze=False,
                             gridspec_kw={"wspace": 0.75 if own_labels else 0.2,
                                          "hspace": 0.9 if own_labels else 0.35})
    norm = matplotlib.colors.Normalize(vmin=chance, vmax=100)
    for idx, ax in enumerate(axes.ravel()):
        ax.set_facecolor(SURFACE)
        if idx >= n:
            ax.axis("off")
            continue
        pnl = panels[idx]
        acc = np.asarray(pnl["acc"], float)
        mesh = ax.pcolormesh(np.ma.masked_invalid(acc), cmap=SEQ, norm=norm, edgecolors=SURFACE, linewidth=1.5)
        _draw_cells(ax, acc, np.asarray(pnl["sig"], bool), norm, fontsize=6.5 if C > 4 else 8,
                    outline_diag=R == C)
        own = pnl.get("labels")
        bottom, left = idx // ncols == nrows - 1 or idx + ncols >= n, idx % ncols == 0
        ax.set_xticks(np.arange(C) + 0.5, own or (col_labels if bottom else []), rotation=35, ha="right",
                      fontsize=7, color=INK_2)
        ax.set_yticks(np.arange(R) + 0.5, own or (row_labels if left else []), fontsize=7, color=INK_2)
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(pnl["title"], fontsize=8.5, color=INK, loc="left")
    fig.supxlabel(xlabel, color=INK_2, fontsize=9)
    fig.supylabel(ylabel, color=INK_2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=10.5, x=0.01, ha="left")
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.01)
    cb.set_ticks([chance, 25, 50, 75, 100], labels=[f"chance {chance:.1f}", "25", "50", "75", "100"])
    cb.ax.tick_params(labelsize=7, colors=INK_2, length=0)
    cb.outline.set_visible(False)
    cb.set_label("accuracy (%)", color=INK_2, fontsize=8)
    if note:
        fig.text(0.01, 0.003, note, fontsize=6.5, color=MUTED, ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def tgm_grid(panels: list[dict], path: Path, title: str, ncols: int, chance: float = 100 / 7,
             note: str = "") -> None:
    """Temporal-generalization maps: accuracy of a decoder trained at each train-window time
    (y, relative to the train behavior's event) tested at each test-window time (x, relative
    to the test behavior's event). The dashed line is 'same offset from the event' — where the
    transfer protocol evaluates; the ring marks the window the protocol actually used.

    panels: [{"acc": (Wa, Wb) %, "t_train": (Wa,) s, "t_test": (Wb,) s, "title", "used": (t, t) | None}]
    """
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols + 0.9, 2.7 * nrows + 0.7), facecolor=SURFACE,
                             squeeze=False)
    norm = matplotlib.colors.Normalize(vmin=chance, vmax=max(60, max(np.nanmax(p["acc"]) for p in panels)))
    for idx, ax in enumerate(axes.ravel()):
        ax.set_facecolor(SURFACE)
        if idx >= n:
            ax.axis("off")
            continue
        p = panels[idx]
        tb, ta = p["t_test"], p["t_train"]
        mesh = ax.pcolormesh(tb, ta, p["acc"], cmap=SEQ, norm=norm, shading="nearest")
        lo, hi = max(ta.min(), tb.min()), min(ta.max(), tb.max())
        if lo < hi:
            ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color=INK_2)
        if p.get("used") is not None:
            ax.plot(*p["used"][::-1], marker="o", ms=7, mfc="none", mec=INK, mew=1.2)
        ax.set_title(p["title"], fontsize=8, color=INK, loc="left")
        ax.tick_params(labelsize=6.5, colors=INK_2, length=2)
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
        if idx % ncols == 0:
            ax.set_ylabel("train window center (s from event)", fontsize=7, color=INK_2)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("test window center (s from event)", fontsize=7, color=INK_2)
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=7, colors=INK_2, length=0)
    cb.outline.set_visible(False)
    cb.set_label("accuracy (%) — floor at chance", color=INK_2, fontsize=8)
    if note:
        fig.text(0.01, 0.003, note, fontsize=6.5, color=MUTED, ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def delta_strip(deltas: dict[str, np.ndarray], highlight: dict[str, np.ndarray], path: Path, title: str) -> None:
    """One row per model: each condition's accuracy change (pp) vs the baseline. Filled dots =
    highlighted conditions (inner speech), hollow = the rest; bar = median; 0 = no change."""
    names = list(deltas)
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(names) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    accent = BLUE_RAMP[7]
    for i, nm in enumerate(names):
        d, hl = np.asarray(deltas[nm]), np.asarray(highlight[nm], bool)
        yj = i + rng.uniform(-0.18, 0.18, len(d))
        ax.scatter(d[~hl], yj[~hl], s=16, facecolors="none", edgecolors=accent, linewidths=0.9, zorder=3)
        ax.scatter(d[hl], yj[hl], s=16, color=accent, edgecolors=SURFACE, linewidths=0.5, zorder=4)
        med = np.median(d)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2, zorder=5)
        ax.text(1.01, i, f"median {med:+.1f}  mean {d.mean():+.1f}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.5, color=INK_2)
    ax.axvline(0, color="#c3c2b7", lw=1, zorder=1)
    ax.set_yticks(range(len(names)), names, fontsize=8.5, color=INK_2)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="x", labelsize=7.5, colors=INK_2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("accuracy change vs baseline (percentage points)", fontsize=8.5, color=INK_2)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=24)
    ax.scatter([], [], s=16, color=accent, label="inner-speech conditions")
    ax.scatter([], [], s=16, facecolors="none", edgecolors=accent, label="other conditions")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              borderaxespad=0.2)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
