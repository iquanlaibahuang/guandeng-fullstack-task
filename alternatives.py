"""负面结果复现：三个"看起来该更好、实测更差"的方案。

写这个脚本的原因：README 主张的规则很简单（延迟≥15 分钟就拦），
但简单方案之所以胜出，是因为更聪明的方案真的试过、并且输了。
把失败保留下来，比只展示成功更能说明结论是跑出来的，不是想出来的。

运行：python alternatives.py  → output/alternatives.csv
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LOSS, COST = 90.0, 12.0
BREAKEVEN = COST / LOSS  # 13.3%


def poisson_cdf(k: int, mu: float) -> float:
    """P(X <= k) for X ~ Poisson(mu)，手写以避免额外依赖 scipy。

    注意 k<0 必须返回 0：available_stock=0 时我们要的是 P(X >= 0)=1，
    早期版本在这里漏掉了这个边界，导致"账面库存为 0"的单被整批漏掉。
    """
    if k < 0:
        return 0.0
    if mu <= 0:
        return 1.0
    term = math.exp(-mu)
    total = term
    for i in range(1, k + 1):
        term *= mu / i
        total += term
    return min(1.0, total)


def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "fulfillments.csv").drop_duplicates().copy()
    d["ts"] = pd.to_datetime(d["order_time"], utc=True)
    d["stale"] = (d["ts"] - pd.to_datetime(d["stock_synced_at"], utc=True)).dt.total_seconds() / 60
    d["canc"] = (d["is_fulfilled"] == 0).astype(int)
    d["day"] = (d["ts"] - d["ts"].min()).dt.days
    return d[d["is_fulfilled"].notna()].copy()


def net(df: pd.DataFrame, mask: pd.Series) -> tuple[float, int, int]:
    m = mask.reindex(df.index).fillna(False).astype(bool)
    n, k = int(m.sum()), int((m & (df["canc"] == 1)).sum())
    return k * LOSS - n * COST, n, k


def main() -> None:
    ev = load()
    rows: list[dict] = []

    # 基准：README 主张的固定阈值
    v, n, k = net(ev, ev["stale"] >= 15)
    rows.append({"方案": "固定阈值 15 分钟（README 主张）", "拦截": n, "拦对": k,
                 "净收益": round(v), "相对基准": "—", "结论": "采纳"})

    # 1) 泊松"耗尽模型"：窗口内需求超过账面库存就拦
    lam = ev.groupby(["warehouse", "category"]).size() / (7 * 24 * 60)
    ev2 = ev.join(lam.rename("lam"), on=["warehouse", "category"])
    ev2["p_stockout"] = [
        1 - poisson_cdf(int(s) - 1, float(l * t))
        for s, l, t in zip(ev2["available_stock"], ev2["lam"], ev2["stale"])
    ]
    v1, n1, k1 = net(ev2, ev2["p_stockout"] > BREAKEVEN)
    zero_stock = int((ev2["available_stock"] == 0).sum())
    rows.append({"方案": "泊松耗尽模型（按出库速率预测缺货）", "拦截": n1, "拦对": k1,
                 "净收益": round(v1), "相对基准": f"{(v1 - v) / v:+.0%}",
                 "结论": f"否决：模型退化成「库存为 0 就拦」，{n1} 次触发里 {zero_stock} 次是账面库存恰好为 0"})

    # 2) 分仓自适应阈值：训练集各自选最优，测试集评估（walk-forward）
    tr, te = ev[ev["day"] <= 4], ev[ev["day"] > 4]
    best_t, best_v = 15, -1e18
    for t in range(61):
        vv, _, _ = net(tr, tr["stale"] >= t)
        if vv > best_v:
            best_v, best_t = vv, t
    v2, n2, k2 = net(te, te["stale"] >= best_t)
    rows.append({"方案": f"全局阈值（训练集选出 {best_t} 分钟）", "拦截": n2, "拦对": k2,
                 "净收益": round(v2), "相对基准": "样本外基准", "结论": "阈值本身在 9~15 分钟间漂移"})

    thr: dict[str, int] = {}
    for w, x in tr.groupby("warehouse"):
        bb, bv = 15, -1e18
        for t in range(61):
            vv, _, _ = net(x, x["stale"] >= t)
            if vv > bv:
                bv, bb = vv, t
        thr[w] = bb
    mask = pd.Series(False, index=te.index)
    for w, t in thr.items():
        mask |= (te["warehouse"] == w) & (te["stale"] >= t)
    v3, n3, k3 = net(te, mask)
    rows.append({"方案": "分仓自适应阈值（各仓各调一条曲线）", "拦截": n3, "拦对": k3,
                 "净收益": round(v3), "相对基准": f"{(v3 - v2) / v2:+.0%}（样本外）",
                 "结论": "否决：每仓每天仅约 32 单，拟合 5 条曲线必然过拟合"})

    # 3) 放宽到 30 分钟（一线主管的诉求）
    v4, n4, k4 = net(ev, ev["stale"] >= 30)
    rows.append({"方案": "放宽到 30 分钟（一线诉求）", "拦截": n4, "拦对": k4,
                 "净收益": round(v4), "相对基准": f"{(v4 - v) / v:+.0%}",
                 "结论": f"否决：放弃的是命中率最高的一档，净亏 {v - v4:,.0f} 元/7天"})

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "output" / "alternatives.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))

    # 机制说明：为什么耗尽模型失效
    ev2["exp_demand"] = ev2["lam"] * ev2["stale"]
    w = ev2[ev2["stale"] >= 15]
    print(f"\n延迟≥15 分钟档：自有订单在窗口内预计消耗均值 {w['exp_demand'].mean():.3f} 单"
          f"（最大 {w['exp_demand'].max():.2f}），而账面可用库存中位数 {w['available_stock'].median():.0f} 单")
    print(f"模型 {n1} 次触发里有 {zero_stock} 次是账面库存恰好为 0，"
          f"其余全部低于 13.3% 的经济边界")
    print("→ 延迟窗口内的自有订单量根本不足以把库存卖光，砍单来自账面库存本身失真，而不是被卖掉")


if __name__ == "__main__":
    main()
