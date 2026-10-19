"""关灯科技 履约缺货分析 —— 全部分析口径的唯一来源。

运行：python analyze.py
产出：output/findings.json（README 中所有数字的来源）+ figures/*.png

口径约定（写进 README，评审可逐条核对）：
  * 去重：完全重复行 58 条按 order_id 去重，保留首条 → 1162 单
  * 时区：order_time 为 +08:00，stock_synced_at 为 UTC(Z)，统一转 UTC 后相减
  * 同步延迟 staleness_min = order_time - stock_synced_at（分钟）
  * 履约状态缺失(is_fulfilled 为空)的 46 单标记为「未回传」，不进入取消率分母
  * 金额：缺货砍单 90 元/次，提前拦截预警 12 元/次（题给口径）
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "fulfillments.csv"
OUT = ROOT / "output"
FIG = ROOT / "figures"
LOSS_PER_CANCEL = 90.0
COST_PER_INTERCEPT = 12.0
STALE_THRESHOLD = 15  # 分钟，主结论的阈值

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA)
    d = raw.drop_duplicates().copy()
    d["order_ts"] = pd.to_datetime(d["order_time"], utc=True)
    d["sync_ts"] = pd.to_datetime(d["stock_synced_at"], utc=True)
    d["staleness_min"] = (d["order_ts"] - d["sync_ts"]).dt.total_seconds() / 60
    d["status"] = np.select(
        [d["is_fulfilled"].isna(), d["is_fulfilled"] == 1],
        ["unreported", "fulfilled"], default="cancelled",
    )
    meta = {
        "raw_rows": int(len(raw)),
        "dup_rows": int(raw.duplicated().sum()),
        "dedup_rows": int(len(d)),
        "unreported": int((d["status"] == "unreported").sum()),
        "date_min": str(d["order_ts"].min()),
        "date_max": str(d["order_ts"].max()),
        "days": int((d["order_ts"].max() - d["order_ts"].min()).days + 1),
        "tz_naive_staleness_mean": round(float(
            (pd.to_datetime(d["order_time"].str[:19]) -
             pd.to_datetime(d["stock_synced_at"].str[:19])).dt.total_seconds().div(60).mean()), 1),
    }
    return d, meta


def policy_table(d: pd.DataFrame) -> pd.DataFrame:
    """给定拦截规则，算「拦多少 / 拦对多少 / 花多少 / 省多少 / 净赚」。"""
    ev = d[d["status"] != "unreported"].copy()
    cancelled = ev["status"] == "cancelled"

    rules = {
        "A 只看库存少 (stock<=5)": ev["available_stock"] <= 5,
        f"B 只看同步旧 (stale>={STALE_THRESHOLD}m)": ev["staleness_min"] >= STALE_THRESHOLD,
        "C 旧且库存少 (stale>=15 & stock<=10)": (ev["staleness_min"] >= STALE_THRESHOLD) & (ev["available_stock"] <= 10),
        "D 超旧 (stale>=30m)": ev["staleness_min"] >= 30,
        "E 旧或库存少 (stale>=15 | stock<=5)": (ev["staleness_min"] >= STALE_THRESHOLD) | (ev["available_stock"] <= 5),
    }
    rows = []
    for name, mask in rules.items():
        n = int(mask.sum())
        caught = int((mask & cancelled).sum())
        hit_rate = caught / n if n else 0.0
        recall = caught / int(cancelled.sum())
        cost = n * COST_PER_INTERCEPT
        saved = caught * LOSS_PER_CANCEL
        rows.append({
            "rule": name, "intercepted": n, "cancels_caught": caught,
            "precision": round(hit_rate, 4), "recall": round(recall, 4),
            "cost": round(cost, 1), "loss_avoided": round(saved, 1),
            "net": round(saved - cost, 1),
        })
    return pd.DataFrame(rows).sort_values("net", ascending=False)


def stale_buckets(d: pd.DataFrame) -> pd.DataFrame:
    ev = d[d["status"] != "unreported"].copy()
    bins = [-0.1, 5, 15, 30, 60, 1e9]
    labels = ["0-5 分钟", "5-15 分钟", "15-30 分钟", "30-60 分钟", "60 分钟以上"]
    ev["bucket"] = pd.cut(ev["staleness_min"], bins=bins, labels=labels)
    g = ev.groupby("bucket", observed=True)["status"].agg(
        orders="count", cancelled=lambda s: int((s == "cancelled").sum()))
    g["cancel_rate"] = (g["cancelled"] / g["orders"]).round(4)
    g["share"] = (g["orders"] / len(ev)).round(4)
    return g.reset_index()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 置信区间——小样本比例不能只看点估计。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def warehouse_view(d: pd.DataFrame) -> pd.DataFrame:
    ev = d[d["status"] != "unreported"]
    rows = []
    for name, x in ev.groupby("warehouse"):
        st = x[x["staleness_min"] >= STALE_THRESHOLD]
        k, n = int((st["status"] == "cancelled").sum()), int(len(st))
        lo, hi = wilson(k, n)
        rows.append({
            "warehouse": name,
            "orders": len(x),
            "cancel_rate": round(float((x["status"] == "cancelled").mean()), 4),
            "stale_share": round(float((x["staleness_min"] >= STALE_THRESHOLD).mean()), 4),
            "median_stale": round(float(x["staleness_min"].median()), 1),
            "cancel_rate_fresh": round(float((x.loc[x["staleness_min"] < STALE_THRESHOLD, "status"] == "cancelled").mean()), 4),
            "stale_n": n, "stale_cancelled": k,
            "cancel_rate_stale": round(k / n, 4) if n else float("nan"),
            "stale_ci_low": round(lo, 3), "stale_ci_high": round(hi, 3),
        })
    return pd.DataFrame(rows).sort_values("cancel_rate", ascending=False)


def warehouse_pnl(d: pd.DataFrame) -> pd.DataFrame:
    """V2 的依据：把 P0 规则按仓拆开，看每个仓是赚是赔。

    单笔期望 = 命中率×90 − 12。为负的仓（本例是华东一仓）应降级为影子模式：
    只记录不拦截，成本为 0，同时继续积累标签以便将来恢复。
    """
    ev = d[d["status"] != "unreported"]
    st = ev[ev["staleness_min"] >= STALE_THRESHOLD]
    g = st.groupby("warehouse")["status"].agg(intercepted="count",
                                              caught=lambda s: int((s == "cancelled").sum()))
    g["hit_rate"] = (g["caught"] / g["intercepted"]).round(4)
    g["cost"] = g["intercepted"] * COST_PER_INTERCEPT
    g["saved"] = g["caught"] * LOSS_PER_CANCEL
    g["net"] = g["saved"] - g["cost"]
    g["net_per_intercept"] = (g["hit_rate"] * LOSS_PER_CANCEL - COST_PER_INTERCEPT).round(1)
    g["verdict"] = np.where(g["net_per_intercept"] > 0, "保持拦截", "降级为影子模式")
    return g.sort_values("net_per_intercept", ascending=False).reset_index()


def sweep(d: pd.DataFrame) -> pd.DataFrame:
    ev = d[d["status"] != "unreported"]
    cancelled = (ev["status"] == "cancelled").to_numpy()
    rows = []
    for t in range(0, 61):
        mask = (ev["staleness_min"] >= t).to_numpy()
        n = int(mask.sum())
        caught = int((mask & cancelled).sum())
        rows.append({"threshold_min": t, "intercepted": n,
                     "net": round(caught * LOSS_PER_CANCEL - n * COST_PER_INTERCEPT, 1)})
    return pd.DataFrame(rows)


def figs(bucket: pd.DataFrame, pol: pd.DataFrame, swp: pd.DataFrame, wh: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    colors = ["#2e7d32", "#66bb6a", "#ef6c00", "#d84315"]
    b = ax.bar(bucket["bucket"], bucket["cancel_rate"] * 100, color=colors, width=0.62)
    for rect, (_, r) in zip(b, bucket.iterrows()):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.8,
                f"{r['cancel_rate']*100:.1f}%\n(n={int(r['orders'])})",
                ha="center", va="bottom", fontsize=8.5)
    ax.axhline(10.66, ls="--", lw=1, color="#555")
    ax.text(len(bucket) - 0.5, 11.4, "整体 10.7%", ha="right", fontsize=8, color="#555")
    ax.set_ylabel("缺货砍单率 (%)")
    ax.set_xlabel("库存同步延迟（下单距最后一次库存同步）")
    ax.set_title("同步延迟跨过 15 分钟，砍单率从 4-5% 跳到 32-35%", fontsize=11)
    ax.set_ylim(0, 45)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "cancel_rate_by_staleness.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    ax.plot(swp["threshold_min"], swp["net"], lw=2, color="#1565c0")
    best = swp.loc[swp["net"].idxmax()]
    ax.scatter([best["threshold_min"]], [best["net"]], color="#c62828", zorder=5)
    ax.annotate(f"最优阈值 {int(best['threshold_min'])} 分钟\n净收益 {best['net']:.0f} 元/7天",
                (best["threshold_min"], best["net"]), textcoords="offset points",
                xytext=(12, -28), fontsize=9, color="#c62828")
    ax.axvline(STALE_THRESHOLD, ls=":", color="#888")
    ax.set_xlabel("拦截阈值：同步延迟 ≥ X 分钟即拦截")
    ax.set_ylabel("净收益（元 / 7 天）")
    ax.set_title("阈值扫描：净收益 = 拦住砍单省 90 元 − 每次拦截花 12 元", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "threshold_sweep.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    x = np.arange(len(wh))
    ax.bar(x, wh["cancel_rate_stale"] * 100, width=0.55, color="#d84315", label="同步延迟≥15分钟")
    ax.bar(x, wh["cancel_rate_fresh"] * 100, width=0.55, color="#2e7d32", label="同步延迟<15分钟")
    for i, (_, r) in enumerate(wh.iterrows()):
        ax.text(i, r["cancel_rate_stale"] * 100 + 1.2,
                f"{r['cancel_rate_stale']*100:.0f}%\n({int(r['stale_cancelled'])}/{int(r['stale_n'])})",
                ha="center", va="bottom", fontsize=7.5, color="#8d2b0b")
    ax.set_xticks(x, wh["warehouse"])
    ax.set_ylabel("缺货砍单率 (%)")
    ax.set_ylim(0, 58)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("华东一仓是例外：同样延迟 23% 的订单，砍单率只有 5.7%", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "warehouse_split.png")
    plt.close(fig)


def main() -> dict:
    d, meta = load()
    ev = d[d["status"] != "unreported"]
    bucket = stale_buckets(d)
    pol = policy_table(d)
    wh = warehouse_view(d)
    swp = sweep(d)
    figs(bucket, pol, swp, wh)

    overall_cancel = float((ev["status"] == "cancelled").mean())
    fresh = ev[ev["staleness_min"] < STALE_THRESHOLD]
    stale = ev[ev["staleness_min"] >= STALE_THRESHOLD]
    best = swp.loc[swp["net"].idxmax()]
    best_rule = pol.iloc[0]
    pnl = warehouse_pnl(d)
    enabled = pnl.loc[pnl["net_per_intercept"] > 0, "warehouse"].tolist()
    disabled = pnl.loc[pnl["net_per_intercept"] <= 0, "warehouse"].tolist()
    mask_v2 = (d["staleness_min"] >= STALE_THRESHOLD) & d["warehouse"].isin(enabled) \
        & (d["status"] != "unreported")
    v2_caught = int((mask_v2 & (d["status"] == "cancelled")).sum())
    v2_n = int(mask_v2.sum())
    v2_net = v2_caught * LOSS_PER_CANCEL - v2_n * COST_PER_INTERCEPT

    findings = {
        "meta": meta,
        "overall": {
            "evaluated_orders": int(len(ev)),
            "cancelled": int((ev["status"] == "cancelled").sum()),
            "cancel_rate": round(overall_cancel, 4),
            "stale_share": round(len(stale) / len(ev), 4),
        },
        "fresh_vs_stale": {
            "fresh_orders": int(len(fresh)),
            "fresh_cancel_rate": round(float((fresh["status"] == "cancelled").mean()), 4),
            "stale_orders": int(len(stale)),
            "stale_cancel_rate": round(float((stale["status"] == "cancelled").mean()), 4),
            "lift": round(float((stale["status"] == "cancelled").mean() /
                                (fresh["status"] == "cancelled").mean()), 1),
            "stale_cancels": int((stale["status"] == "cancelled").sum()),
            "stale_cancel_share": round(float((stale["status"] == "cancelled").sum() /
                                              (ev["status"] == "cancelled").sum()), 4),
        },
        "staleness_buckets": bucket.to_dict(orient="records"),
        "policies": pol.to_dict(orient="records"),
        "best_threshold": {"threshold_min": int(best["threshold_min"]), "net_per_7d": float(best["net"])},
        "warehouse": wh.to_dict(orient="records"),
        "recommended": {
            "rule": best_rule["rule"],
            "intercepted": int(best_rule["intercepted"]),
            "cancels_caught": int(best_rule["cancels_caught"]),
            "precision": best_rule["precision"],
            "recall": best_rule["recall"],
            "cost_per_7d": best_rule["cost"],
            "loss_avoided_per_7d": best_rule["loss_avoided"],
            "net_per_7d": best_rule["net"],
            "net_per_day": round(best_rule["net"] / meta["days"], 1),
            "net_per_year": round(best_rule["net"] / meta["days"] * 365, 1),
        },
        "counterfactual_no_stale": {
            "explanation": "若把同步延迟压到 15 分钟以内，延迟组砍单率会回落到新鲜组水平",
            "cancels_avoided": round(float(stale["status"].eq("cancelled").sum() -
                                           len(stale) * (fresh["status"] == "cancelled").mean()), 1),
            "value_per_7d": round(float((stale["status"].eq("cancelled").sum() -
                                         len(stale) * (fresh["status"] == "cancelled").mean())) * LOSS_PER_CANCEL, 1),
        },
        "non_drivers": {
            "available_stock_corr": round(float(np.corrcoef(ev["available_stock"], (ev["status"] == "cancelled"))[0, 1]), 4),
            "daily_orders_corr": round(float(np.corrcoef(ev["daily_orders"], (ev["status"] == "cancelled"))[0, 1]), 4),
            "staleness_corr": round(float(np.corrcoef(ev["staleness_min"], (ev["status"] == "cancelled"))[0, 1]), 4),
        },
        "v2_warehouse_switch": {
            "explanation": "P0 规则按仓拆开算单笔期望，把为负的仓降级为影子模式（只记录不拦截、成本 0）",
            "enabled": enabled,
            "shadow": disabled,
            "intercepted": v2_n,
            "cancels_caught": v2_caught,
            "net_per_7d": round(float(v2_net), 1),
            "uplift_vs_v1": round(float(v2_net) - best_rule["net"], 1),
            "uplift_pct": round((float(v2_net) - best_rule["net"]) / best_rule["net"] * 100, 1),
        },
        "warehouse_pnl": pnl.to_dict(orient="records"),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    bucket.to_csv(OUT / "cancel_rate_by_staleness.csv", index=False, encoding="utf-8-sig")
    pol.to_csv(OUT / "policy_compare.csv", index=False, encoding="utf-8-sig")
    wh.to_csv(OUT / "warehouse_view.csv", index=False, encoding="utf-8-sig")
    swp.to_csv(OUT / "threshold_sweep.csv", index=False, encoding="utf-8-sig")
    pnl.to_csv(OUT / "warehouse_pnl.csv", index=False, encoding="utf-8-sig")
    return findings


if __name__ == "__main__":
    f = main()
    print(json.dumps({k: f[k] for k in ("meta", "overall", "fresh_vs_stale", "recommended", "non_drivers")},
                     ensure_ascii=False, indent=2))
    print("\n=== buckets ===\n", pd.DataFrame(f["staleness_buckets"]).to_string(index=False))
    print("\n=== policies ===\n", pd.DataFrame(f["policies"]).to_string(index=False))
    print("\n=== warehouse ===\n", pd.DataFrame(f["warehouse"]).to_string(index=False))
    print("\n=== best threshold ===", f["best_threshold"])
    print("=== counterfactual ===", f["counterfactual_no_stale"])
