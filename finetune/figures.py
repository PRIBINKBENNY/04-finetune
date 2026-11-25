"""Dataset overview figure for the README:  python -m finetune.figures  ->  docs/dataset.png"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .data import build


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
                         "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False})
    train, test = build(2000, 200)
    rows = train + test
    T = [r["target"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    blue, orange = "#264bd8", "#e76f51"

    cur = Counter(t["currency"] or "none" for t in T)
    axes[0, 0].bar(cur.keys(), cur.values(), color=blue)
    axes[0, 0].set_title("Currency (none = no salary given)")

    present = {f: sum(t[f] not in (None, [], "") for t in T) / len(T) * 100
               for f in ["location", "salary_min", "salary_max", "currency"]}
    present["remote=true"] = sum(t["remote"] for t in T) / len(T) * 100
    axes[0, 1].barh(list(present), list(present.values()), color=blue)
    axes[0, 1].set_title("Field present (% of examples)")
    axes[0, 1].set_xlim(0, 100)
    axes[0, 1].invert_yaxis()

    ns = Counter(len(t["skills"]) for t in T)
    axes[0, 2].bar([str(k) for k in sorted(ns)], [ns[k] for k in sorted(ns)], color=blue)
    axes[0, 2].set_title("Skills per post")

    sk = Counter(s for t in T for s in t["skills"]).most_common(12)
    axes[1, 0].barh([s for s, _ in sk], [c for _, c in sk], color=orange)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("Most common skills")

    lens = [len(r["text"]) for r in rows]
    axes[1, 1].hist(lens, bins=30, color=orange)
    axes[1, 1].set_title("Post length (characters)")

    ax = axes[1, 2]
    ax.axis("off")
    ex = test[3]
    ax.text(0, 1, "Example (test set)", weight="bold", va="top", fontsize=11, transform=ax.transAxes)
    ax.text(0, 0.86, "input:", color="#555", va="top", transform=ax.transAxes)
    ax.text(0, 0.78, _wrap(ex["text"], 52), va="top", family="DejaVu Sans Mono", fontsize=8.5, transform=ax.transAxes)
    ax.text(0, 0.42, "target:", color="#555", va="top", transform=ax.transAxes)
    ax.text(0, 0.34, _wrap(json.dumps(ex["target"]), 52), va="top", family="DejaVu Sans Mono", fontsize=8.5,
            transform=ax.transAxes)
    for a in axes.ravel()[:5]:
        a.grid(alpha=0.2)
    fig.suptitle(f"Synthetic job-post extraction dataset: {len(train):,} train / {len(test)} test, 6 templates, 4 currencies",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/dataset.png", dpi=100, facecolor="white")
    print("wrote docs/dataset.png")
    return 0


def _wrap(s: str, n: int) -> str:
    out, line = [], ""
    for w in s.replace("\n", " ⏎ ").split(" "):
        if len(line) + len(w) + 1 > n:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    out.append(line)
    return "\n".join(out).replace("$", r"\$")  # literal dollar signs, not matplotlib maths


if __name__ == "__main__":
    raise SystemExit(main())
