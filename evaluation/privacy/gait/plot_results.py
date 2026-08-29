"""
plot_results.py -- figures for the anti-reID pose evaluation.
Reads reports/reid/{BASELINE_tm12,REDESIGN_tm12,SWEEP_pareto,TM3_*}.json and writes
  reports/reid/fig_threatmodel.png  -- the pseudonymity story (v1 fails, v2 fixes)
  reports/reid/fig_pareto.png       -- privacy<->utility Pareto (+ TM3 overlay if present)
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "reports", "reid")
CHANCE = 2.0

def load(name):
    p = os.path.join(RD, name)
    return json.load(open(p)) if os.path.exists(p) else None

INK = "#1b1b1f"; GRID = "#d9d9de"
C_TM1 = "#3b6fb0"; C_TM2 = "#c1543a"; C_CANON = "#4a9d7f"; C_CHANCE = "#8a8a92"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                     "figure.dpi": 130, "savefig.dpi": 130})


def fig_threatmodel():
    red = load("REDESIGN_tm12.json")
    if not red:
        return
    rows = red["rows"]
    order = ["raw", "v1-ours", "v2-canon", "v2-canon+cad", "v2-canon+ang", "v2-full"]
    order = [o for o in order if o in rows]
    labels = {"raw": "raw\n(no anon)", "v1-ours": "v1\n(old)", "v2-canon": "v2\ncanon",
              "v2-canon+cad": "v2\n+cad", "v2-canon+ang": "v2\n+ang", "v2-full": "v2\nfull"}
    tm1 = [rows[o]["TM1"]["NM"] * 100 for o in order]
    tm2id = [rows[o]["TM2id"]["NM"] * 100 for o in order]
    tm2sq = [rows[o]["TM2sq"]["NM"] * 100 for o in order]
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.bar(x - w, tm1, w, label="TM1  gallery clean / probe anon", color=C_TM1)
    ax.bar(x, tm2id, w, label="TM2-id  both anon, per-identity seed", color=C_TM2)
    ax.bar(x + w, tm2sq, w, label="TM2-seq  both anon, per-sequence seed", color=C_CANON)
    ax.axhline(CHANCE, color=C_CHANCE, ls="--", lw=1.2)
    ax.text(len(order) - 0.5, CHANCE + 1.5, "chance = 2%", color=C_CHANCE, ha="right", fontsize=9)
    ax.annotate("v1 is pseudonymity:\nTM2-id stays at raw", xy=(1, tm2id[1]), xytext=(1.5, 70),
                fontsize=8.5, color=C_TM2, arrowprops=dict(arrowstyle="->", color=C_TM2))
    ax.set_xticks(x); ax.set_xticklabels([labels[o] for o in order])
    ax.set_ylabel("adversary rank-1 (NM), % - lower = more private")
    ax.set_title("Gait re-ID vs. anonymizer design (frozen GaitGraph adversary)")
    ax.set_ylim(0, 100); ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(RD, "fig_threatmodel.png")); plt.close(fig)
    print("wrote fig_threatmodel.png")


def fig_pareto():
    sw = load("SWEEP_pareto.json")
    if not sw:
        return
    rows = sw["rows"]
    ladder = [k for k in ["L0 canon", "L1", "L2", "L3", "L4", "L5 aggressive"] if k in rows]
    mp = [rows[k]["util"][0] for k in ladder]
    tm1 = [rows[k]["TM1"]["NM"] * 100 for k in ladder]
    tm2 = [rows[k]["TM2"]["NM"] * 100 for k in ladder]
    # optional TM3 adaptive overlay
    tm3 = {}
    for f in glob.glob(os.path.join(RD, "TM3_*.json")):
        d = json.load(open(f)); tm3[d["config"]] = d["final"]["NM"] * 100
    cfgmap = {"L0 canon": "v2-canon", "L2": "v2-full", "L3": "L3", "L4": "L4", "L5 aggressive": "L5"}

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(mp, tm1, "-o", color=C_TM1, label="TM1 frozen (clean/anon)")
    ax.plot(mp, tm2, "-s", color=C_TM2, label="TM2 frozen (anon/anon)")
    for k, x, y in zip(ladder, mp, tm2):
        ax.annotate(k.split()[0], (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8)
    tx = [mp[i] for i, k in enumerate(ladder) if cfgmap.get(k) in tm3]
    ty = [tm3[cfgmap[k]] for k in ladder if cfgmap.get(k) in tm3]
    if tx:
        ax.plot(tx, ty, "-^", color="#7a4fb0", label="TM3 ADAPTIVE (retrained on anon)")
    ax.axhline(CHANCE, color=C_CHANCE, ls="--", lw=1.2)
    ax.text(max(mp), CHANCE + 0.8, "chance = 2%", color=C_CHANCE, ha="right", fontsize=9)
    ax.set_xlabel("utility cost - MPJPE vs original (px, 320px frame)")
    ax.set_ylabel("adversary rank-1 (NM), % - lower = more private")
    ax.set_title("Privacy vs. utility Pareto (pose_anon_v2, per-sequence seeding)")
    ax.legend(fontsize=8.5, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color=GRID, lw=0.6)
    ax.set_ylim(0, max(25, max(tm1) + 3))
    fig.tight_layout(); fig.savefig(os.path.join(RD, "fig_pareto.png")); plt.close(fig)
    print("wrote fig_pareto.png" + (" (with TM3 overlay)" if tx else " (frozen only)"))


if __name__ == "__main__":
    fig_threatmodel()
    fig_pareto()
