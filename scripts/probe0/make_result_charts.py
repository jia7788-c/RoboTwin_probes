# 生成 README 结果图（亮/暗两套 SVG）。用法: python scripts/probe0/make_result_charts.py assets/probe0
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
out = sys.argv[1]
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["axes.unicode_minus"] = False

THEMES = {
 "light": dict(surface="#ffffff", t1="#0b0b0b", t2="#52514e", grid="#e4e3df",
               s=["#2a78d6","#eb6834","#1baf7a","#eda100"], na="#b8b7b0"),
 "dark":  dict(surface="#0d1117", t1="#f0efec", t2="#c3c2b7", grid="#30302d",
               s=["#3987e5","#d95926","#199e70","#c98500"], na="#6b6a64"),
}
models = ["AE-only LoRA","Vision Encoder + AE","Default LoRA","Full Finetune BS32"]
tasks = ["handover_block","lift_pot","put_object_cabinet","place_empty_cup","click_alarmclock"]
clean = np.array([[7,15,6,17,53],[46,98,36,93,73],[51,96,40,95,56],[70,93,37,88,58]])
avg = clean.mean(1)

def style(ax, th):
    ax.set_facecolor(th["surface"])
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(th["t2"])
    ax.tick_params(colors=th["t2"], labelsize=10, length=0)
    ax.yaxis.grid(True, color=th["grid"], lw=0.8); ax.set_axisbelow(True)

def bar(ax, x, h, w, c, th):
    ax.bar(x, h, w, color=c, edgecolor=th["surface"], linewidth=1.5)

for mode, th in THEMES.items():
    # ---- 1. clean SR grouped bar ----
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=100)
    fig.patch.set_facecolor(th["surface"]); style(ax, th)
    labels = tasks + ["五任务平均"]
    x = np.arange(6) + np.array([0,0,0,0,0,0.45])
    w = 0.2
    hs=[]
    for i,m in enumerate(models):
        vals = list(clean[i]) + [avg[i]]
        hs.append(ax.bar(x + (i-1.5)*w, vals, w, color=th["s"][i], edgecolor=th["surface"], linewidth=1.5, label=m))
        # label only the average group
        ax.text(x[-1]+(i-1.5)*w, avg[i]+1.5, f"{avg[i]:.1f}", ha="center", va="bottom",
                fontsize=9, color=th["t1"], fontweight="bold")
    ax.axvspan(x[-1]-0.5, x[-1]+0.5, color=th["grid"], alpha=0.35, zorder=0, lw=0)
    ax.set_xticks(x, labels, fontsize=10.5, color=th["t1"])
    ax.set_ylim(0,105); ax.set_yticks(range(0,101,20)); ax.set_ylabel("成功率 SR (%)", color=th["t2"])
    ax.set_title("原五任务联合模型 · Clean 成功率（每格 100 回合，H=50, K=50）", loc="left",
                 color=th["t1"], fontsize=13, pad=26)
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0,1.09), ncol=4, frameon=False,
              fontsize=10, labelcolor=th["t1"], handlelength=1.2)
    fig.tight_layout(); fig.savefig(f"{out}/clean_sr_{mode}.svg", facecolor=th["surface"]); plt.close(fig)

    # ---- 2. clean vs DR ----
    dr = [6.0, 12.0, None, 31.0]
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=100)
    fig.patch.set_facecolor(th["surface"]); style(ax, th)
    ax.yaxis.grid(False); ax.xaxis.grid(True, color=th["grid"], lw=0.8)
    ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(True); ax.spines["left"].set_color(th["t2"])
    y = np.arange(4)[::-1]
    for i in range(4):
        c = th["s"][i]
        ax.barh(y[i]+0.19, avg[i], 0.36, color=c, edgecolor=th["surface"], lw=1.5)
        ax.text(avg[i]+1, y[i]+0.19, f"Clean {avg[i]:.1f}%", va="center", fontsize=9.5, color=th["t1"])
        if dr[i] is None:
            ax.text(1, y[i]-0.19, "DR 仅完成 3/5 任务，不计均值", va="center", fontsize=9.5, color=th["t2"], style="italic")
        else:
            ax.barh(y[i]-0.19, dr[i], 0.36, color=c, alpha=0.4, hatch="////", edgecolor=c, lw=0)
            ax.text(dr[i]+1, y[i]-0.19, f"DR {dr[i]:.1f}%", va="center", fontsize=9.5, color=th["t1"])
    ax.set_yticks(y, models, color=th["t1"], fontsize=10.5)
    ax.set_xlim(0,100); ax.set_xlabel("五任务平均 SR (%)", color=th["t2"])
    ax.set_title("域随机化（demo_randomized）：深色 = Clean，浅色 = DR", loc="left", color=th["t1"], fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{out}/clean_vs_dr_{mode}.svg", facecolor=th["surface"]); plt.close(fig)

    # ---- 3. heatmap per task (sequential, single hue) ----
    from matplotlib.colors import LinearSegmentedColormap
    base = th["s"][0]
    cmap = LinearSegmentedColormap.from_list("b", [th["surface"], base]) 
    fig, ax = plt.subplots(figsize=(8.5, 3.3), dpi=100)
    fig.patch.set_facecolor(th["surface"]); ax.set_facecolor(th["surface"])
    ax.imshow(clean, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(4):
        for j in range(5):
            v = clean[i,j]; best = v == clean[:,j].max()
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=11,
                    color=("#ffffff" if v>=80 else th["t1"]), fontweight=("bold" if best else "normal"))
    ax.set_xticks(range(5), tasks, color=th["t1"], fontsize=9.5)
    ax.set_yticks(range(4), models, color=th["t1"], fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5,5,1), minor=True); ax.set_yticks(np.arange(-.5,4,1), minor=True)
    ax.grid(which="minor", color=th["surface"], lw=2.5); ax.tick_params(which="minor", length=0)
    ax.set_title("逐任务成功率热力图（%，加粗 = 该任务最高）", loc="left", color=th["t1"], fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{out}/clean_heatmap_{mode}.svg", facecolor=th["surface"]); plt.close(fig)

    # ---- 4. long tasks ----
    lm = ["AE-only LoRA","Default LoRA","Full Finetune"]; lv=[2.4,43.2,27.2]; lc=[th["s"][0],th["s"][2],th["s"][3]]
    fig, ax = plt.subplots(figsize=(7, 2.4), dpi=100)
    fig.patch.set_facecolor(th["surface"]); style(ax, th)
    ax.yaxis.grid(False); ax.xaxis.grid(True, color=th["grid"], lw=.8); ax.spines["bottom"].set_visible(False)
    yy=np.arange(3)[::-1]
    ax.barh(yy, lv, 0.55, color=lc, edgecolor=th["surface"], lw=1.5)
    for k in range(3): ax.text(lv[k]+0.8, yy[k], f"{lv[k]:.1f}%", va="center", fontsize=10, color=th["t1"])
    ax.set_yticks(yy, lm, color=th["t1"]); ax.set_xlim(0,60); ax.set_xlabel("长五任务平均 SR (%)", color=th["t2"])
    ax.set_title("历史扩展 · 长五任务（batch 2，150k 步）", loc="left", color=th["t1"], fontsize=12)
    fig.tight_layout(); fig.savefig(f"{out}/long_tasks_{mode}.svg", facecolor=th["surface"]); plt.close(fig)
print("ok")
