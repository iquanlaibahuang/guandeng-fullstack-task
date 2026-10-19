"""履约缺货看板：一个阈值滑杆，实时算"省多少 / 花多少 / 净赚多少"。

运行：python app.py  →  http://127.0.0.1:8000
依赖：fastapi, uvicorn, pandas
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parent
LOSS, COST = 90.0, 12.0

app = FastAPI(title="履约缺货归因看板")

_cache: dict[str, pd.DataFrame] = {}


def orders() -> pd.DataFrame:
    if "d" not in _cache:
        d = pd.read_csv(ROOT / "data" / "fulfillments.csv").drop_duplicates().copy()
        d["staleness_min"] = (
            pd.to_datetime(d["order_time"], utc=True) - pd.to_datetime(d["stock_synced_at"], utc=True)
        ).dt.total_seconds() / 60
        d = d[d["is_fulfilled"].notna()]              # 状态未回传不进入分母
        d["cancelled"] = (d["is_fulfilled"] == 0).astype(int)
        _cache["d"] = d
    return _cache["d"]


@app.get("/api/findings")
def findings() -> dict:
    p = ROOT / "output" / "findings.json"
    if not p.exists():
        raise HTTPException(503, "请先运行 python analyze.py")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/simulate")
def simulate(threshold: float = 15.0, stock_max: float | None = None) -> dict:
    """给定拦截阈值，返回这笔生意的账。"""
    d = orders()
    mask = d["staleness_min"] >= threshold
    if stock_max is not None:
        mask &= d["available_stock"] <= stock_max
    n = int(mask.sum())
    caught = int(d.loc[mask, "cancelled"].sum())
    total_cancels = int(d["cancelled"].sum())
    cost, saved = n * COST, caught * LOSS
    return {
        "threshold_min": threshold,
        "stock_max": stock_max,
        "evaluated_orders": int(len(d)),
        "intercepted": n,
        "intercept_share": round(n / len(d), 4),
        "cancels_caught": caught,
        "precision": round(caught / n, 4) if n else 0.0,
        "recall": round(caught / total_cancels, 4),
        "breakeven_precision": round(COST / LOSS, 4),
        "cost": cost, "loss_avoided": saved,
        "net": round(saved - cost, 1),
        "net_per_day": round((saved - cost) / 7, 1),
        "verdict": "赚" if saved - cost > 0 else "亏",
    }


@app.get("/figures/{name}")
def figure(name: str) -> FileResponse:
    p = (ROOT / "figures" / name).resolve()
    if p.parent != (ROOT / "figures").resolve() or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "dashboard.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
