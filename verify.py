"""独立重算 README 里的每一条数字，和 findings.json 对账。

刻意不复用 analyze.py 的任何函数——避免"用同一段代码验证自己"。
运行：python verify.py  → 全部 PASS 才说明 README 没写错。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LOSS, COST = 90.0, 12.0


def recompute() -> dict:
    raw = pd.read_csv(ROOT / "data" / "fulfillments.csv")
    d = raw.drop_duplicates().copy()
    t_order = pd.to_datetime(d["order_time"], utc=True)
    t_sync = pd.to_datetime(d["stock_synced_at"], utc=True)
    d["stale"] = (t_order - t_sync).dt.total_seconds().div(60)
    ev = d[d["is_fulfilled"].notna()]
    cancelled = ev["is_fulfilled"] == 0

    stale = ev["stale"] >= 15
    fresh = ev["stale"] < 15
    n_stale, n_fresh = int(stale.sum()), int(fresh.sum())
    k_stale, k_fresh = int(cancelled[stale].sum()), int(cancelled[fresh].sum())

    intercept = n_stale
    caught = k_stale
    net = caught * LOSS - intercept * COST

    naive = int((ev["available_stock"] <= 5).sum())
    naive_caught = int((cancelled & (ev["available_stock"] <= 5)).sum())

    wh = ev.assign(stale=stale, cancelled=cancelled).groupby("warehouse").apply(
        lambda x: pd.Series({
            "n": int(x["stale"].sum()),
            "k": int((x["stale"] & x["cancelled"]).sum()),
        }), include_groups=False)

    # V2：按仓算单笔期望，只对期望为正的仓保持拦截
    pnl = {w: r["k"] * LOSS - r["n"] * COST for w, r in wh.iterrows()}
    enabled = [w for w, r in wh.iterrows() if r["k"] * LOSS - r["n"] * COST > 0]
    v2_mask = (ev["stale"] >= 15) & ev["warehouse"].isin(enabled)
    v2_n, v2_k = int(v2_mask.sum()), int((v2_mask & cancelled).sum())

    return {
        "去重后订单数": (len(d), 1162),
        "有效样本(排除未回传)": (len(ev), 1116),
        "整体砍单率": (round(float(cancelled.mean()), 4), 0.1066),
        "延迟组砍单率": (round(k_stale / n_stale, 4), 0.3294),
        "新鲜组砍单率": (round(k_fresh / n_fresh, 4), 0.0417),
        "延迟组占比": (round(n_stale / len(ev), 4), 0.2258),
        "延迟组占砍单比": (round(k_stale / int(cancelled.sum()), 4), 0.6975),
        "规则B净收益": (round(net, 1), 4446.0),
        "规则B拦截数": (intercept, 252),
        "规则B拦对砍单": (caught, 83),
        "直觉规则A净收益": (round(naive_caught * LOSS - naive * COST, 1), 84.0),
        "华东一仓延迟组砍单率": (round(float(wh.loc["华东一仓", "k"]) / float(wh.loc["华东一仓", "n"]), 4), 0.0566),
        "华东一仓延迟组样本": (int(wh.loc["华东一仓", "n"]), 53),
        "华东一仓P0净收益": (round(float(pnl["华东一仓"]), 1), -366.0),
        "华东一仓单笔期望": (round((float(wh.loc["华东一仓", "k"]) / float(wh.loc["华东一仓", "n"])) * LOSS - COST, 1), -6.9),
        "V2 拦截数": (v2_n, 199),
        "V2 拦对砍单": (v2_k, 80),
        "V2 净收益": (round(v2_k * LOSS - v2_n * COST, 1), 4812.0),
    }


def main() -> int:
    print(f"{'断言':<22}{'重算值':>12}{'README值':>12}   结果")
    print("-" * 58)
    ok = True
    for name, (got, want) in recompute().items():
        passed = (abs(float(got) - float(want)) <= max(0.001, abs(float(want)) * 0.001))
        ok &= passed
        print(f"{name:<22}{str(got):>12}{str(want):>12}   {'PASS' if passed else 'FAIL'}")

    p = ROOT / "output" / "findings.json"
    if p.exists():
        f = json.loads(p.read_text(encoding="utf-8"))
        match = abs(f["recommended"]["net_per_7d"] - 4446.0) < 1
        ok &= match
        print(f"{'findings.json 对账':<22}{'':>12}{'':>12}   {'PASS' if match else 'FAIL'}")
    print("-" * 58)
    print("全部一致：README 中的数字可由 data/fulfillments.csv 复现" if ok else "存在不一致，请检查")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
