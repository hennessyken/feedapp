"""Generate HTML strategy report with sortable tables and colour coding.

Usage:
    from report_html import generate_html_report
    generate_html_report(results, benchmark_results, ml_report, signals_count, out_path)
"""

from typing import Any, Dict, List, Optional
from strategy_analyzer import StrategyResult, BenchmarkResult


def _colour(val: float, thresholds: tuple = (-1, 0, 2)) -> str:
    """Return CSS colour based on value."""
    if val <= thresholds[0]:
        return "#e74c3c"   # red
    if val <= thresholds[1]:
        return "#e67e22"   # orange
    if val <= thresholds[2]:
        return "#2ecc71"   # green
    return "#27ae60"       # dark green


def _pct(val: float, dp: int = 2) -> str:
    return f"{val:+.{dp}f}%"


def _stop_str(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val*100:.0f}%"


def generate_html_report(
    results: List[StrategyResult],
    benchmark_results: List[BenchmarkResult],
    ml_report: Dict[str, Any],
    signals_count: int,
    out_path: str = "strategy_report.html",
    fundamentals_summary: Optional[Dict] = None,
) -> str:
    """Generate a self-contained HTML report and return the path."""

    # ----- Top strategies by different metrics -----
    top_sharpe = sorted(results, key=lambda r: r.sharpe, reverse=True)[:30]
    top_winrate = sorted(
        [r for r in results if r.trades >= 10],
        key=lambda r: r.win_rate, reverse=True,
    )[:30]
    top_avg = sorted(
        [r for r in results if r.trades >= 10],
        key=lambda r: r.avg_return, reverse=True,
    )[:30]
    top_total = sorted(
        [r for r in results if r.trades >= 20],
        key=lambda r: r.total_return, reverse=True,
    )[:30]

    def _strat_table(strats: List[StrategyResult], sort_col: str) -> str:
        rows = []
        for i, r in enumerate(strats, 1):
            avg_col = _colour(r.avg_return)
            wr_col = _colour(r.win_rate, (30, 45, 60))
            rows.append(f"""<tr>
                <td>{i}</td>
                <td>{r.filter_name}</td>
                <td>{r.hold_days}d</td>
                <td>{_stop_str(r.stop_loss_pct)}</td>
                <td>{r.trades}</td>
                <td style="color:{wr_col}">{r.win_rate:.1f}%</td>
                <td style="color:{avg_col}">{_pct(r.avg_return)}</td>
                <td>{_pct(r.median_return)}</td>
                <td style="color:{_colour(r.total_return, (-50, 0, 50))}">{_pct(r.total_return, 1)}</td>
                <td>{r.sharpe:.2f}</td>
                <td style="color:#2ecc71">{_pct(r.best)}</td>
                <td style="color:#e74c3c">{_pct(r.worst)}</td>
                <td>{_pct(r.max_drawdown, 1)}</td>
            </tr>""")
        return "\n".join(rows)

    # ----- Benchmark table -----
    bm_rows = []
    for bm in sorted(benchmark_results, key=lambda b: b.hold_days):
        exc_col = _colour(bm.excess_return)
        bm_rows.append(f"""<tr>
            <td>{bm.hold_days}d</td>
            <td>{bm.signal_trades}</td>
            <td style="color:{_colour(bm.signal_avg_return)}">{_pct(bm.signal_avg_return)}</td>
            <td>{bm.signal_win_rate:.1f}%</td>
            <td>{_pct(bm.signal_total_return, 1)}</td>
            <td>{_pct(bm.spy_avg_return)}</td>
            <td style="color:{exc_col};font-weight:bold">{_pct(bm.excess_return)}</td>
        </tr>""")

    # ----- ML summary -----
    ml_html = ""
    if ml_report and "error" not in ml_report:
        bg = ml_report.get("best_global", {})
        segments = ml_report.get("segments", [])
        ml_seg_rows = []
        for seg in sorted(segments, key=lambda s: s.get("cv_auc_roc", 0), reverse=True):
            auc_col = _colour(seg.get("cv_auc_roc", 0.5), (0.45, 0.5, 0.55))
            ml_seg_rows.append(f"""<tr>
                <td>{seg.get('segment', '')}</td>
                <td>{seg.get('hold_days', '')}d</td>
                <td>{_stop_str(seg.get('stop_loss_pct'))}</td>
                <td>{seg.get('n_signals', '')}</td>
                <td style="color:{auc_col}">{seg.get('cv_auc_roc', 0):.3f}</td>
                <td>{seg.get('cv_f1', 0):.3f}</td>
                <td>{seg.get('baseline_win_rate', 0):.1f}%</td>
                <td>{_pct(seg.get('baseline_avg_return', 0))}</td>
            </tr>""")

        ml_html = f"""
        <h2>🤖 ML Classifier (XGBoost)</h2>
        <div class="summary-box">
            Best global: hold={bg.get('hold_days','')}d stop={_stop_str(bg.get('stop_loss_pct'))} |
            AUC={bg.get('cv_auc_roc',0):.3f} F1={bg.get('cv_f1',0):.3f} |
            Walk-forward: {bg.get('n_walk_forward_windows', 0)} windows
        </div>
        <table>
            <thead><tr>
                <th>Segment</th><th>Hold</th><th>Stop</th><th>Signals</th>
                <th>AUC ↓</th><th>F1</th><th>Base Win%</th><th>Base Avg</th>
            </tr></thead>
            <tbody>{"".join(ml_seg_rows)}</tbody>
        </table>
        """

    # ----- Fundamentals summary -----
    fund_html = ""
    if fundamentals_summary:
        fund_html = f"""
        <h2>📋 Fundamentals Coverage</h2>
        <div class="summary-box">
            {fundamentals_summary.get('total', 0)} tickers with data |
            {fundamentals_summary.get('with_sector', 0)} with sector |
            Top sectors: {', '.join(f"{s[0]} ({s[1]})" for s in fundamentals_summary.get('top_sectors', [])[:5])}
        </div>
        """

    th = """<th>Rank</th><th>Filter</th><th>Hold</th><th>Stop</th>
            <th>Trades</th><th>Win%</th><th>Avg Ret</th><th>Med Ret</th>
            <th>Total Ret</th><th>Sharpe</th><th>Best</th><th>Worst</th><th>MaxDD</th>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Strategy Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, 'Segoe UI', Roboto, monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
    h1 {{ color: #00d4ff; margin-bottom: 5px; }}
    h2 {{ color: #00d4ff; margin: 30px 0 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }}
    .meta {{ color: #888; margin-bottom: 20px; }}
    .summary-box {{ background: #16213e; padding: 12px 16px; border-radius: 6px; margin-bottom: 15px; border-left: 3px solid #00d4ff; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 13px; }}
    th {{ background: #16213e; color: #00d4ff; padding: 8px 10px; text-align: left; position: sticky; top: 0; cursor: pointer; }}
    th:hover {{ background: #1a3a5c; }}
    td {{ padding: 6px 10px; border-bottom: 1px solid #2a2a3e; }}
    tr:hover {{ background: #16213e; }}
    tr:nth-child(even) {{ background: #1e1e36; }}
    .tab-container {{ display: flex; gap: 5px; margin-bottom: 0; }}
    .tab {{ padding: 8px 16px; background: #16213e; border: 1px solid #333; border-bottom: none;
            border-radius: 6px 6px 0 0; cursor: pointer; color: #888; }}
    .tab.active {{ background: #1a1a2e; color: #00d4ff; border-color: #00d4ff; border-bottom: 1px solid #1a1a2e; }}
    .tab-panel {{ display: none; border: 1px solid #333; border-radius: 0 6px 6px 6px; padding: 10px; }}
    .tab-panel.active {{ display: block; }}
    .good {{ color: #2ecc71; }}
    .bad {{ color: #e74c3c; }}
</style>
</head>
<body>

<h1>📊 Strategy Analysis Report</h1>
<p class="meta">Signals: {signals_count} | Strategies tested: {len(results)} | Generated by feedapp optimizer</p>

{fund_html}

<h2>📈 Benchmark: All Signals vs SPY</h2>
<table>
    <thead><tr>
        <th>Hold Period</th><th>Trades</th><th>Signal Avg</th><th>Signal Win%</th>
        <th>Signal Total</th><th>SPY Avg</th><th>Excess Return</th>
    </tr></thead>
    <tbody>{"".join(bm_rows)}</tbody>
</table>

<h2>🏆 Top Strategies</h2>

<div class="tab-container">
    <div class="tab active" onclick="showTab(event, 'sharpe')">By Sharpe</div>
    <div class="tab" onclick="showTab(event, 'winrate')">By Win Rate</div>
    <div class="tab" onclick="showTab(event, 'avgret')">By Avg Return</div>
    <div class="tab" onclick="showTab(event, 'totalret')">By Total Return</div>
</div>

<div id="sharpe" class="tab-panel active">
<table><thead><tr>{th}</tr></thead>
<tbody>{_strat_table(top_sharpe, 'sharpe')}</tbody></table>
</div>

<div id="winrate" class="tab-panel">
<table><thead><tr>{th}</tr></thead>
<tbody>{_strat_table(top_winrate, 'win_rate')}</tbody></table>
</div>

<div id="avgret" class="tab-panel">
<table><thead><tr>{th}</tr></thead>
<tbody>{_strat_table(top_avg, 'avg_return')}</tbody></table>
</div>

<div id="totalret" class="tab-panel">
<table><thead><tr>{th}</tr></thead>
<tbody>{_strat_table(top_total, 'total_return')}</tbody></table>
</div>

{ml_html}

<h2>📋 All Strategies ({len(results)} total)</h2>
<input type="text" id="filterInput" placeholder="Filter by name..."
    style="background:#16213e; color:#e0e0e0; border:1px solid #333; padding:8px 12px;
    border-radius:4px; width:300px; margin-bottom:10px;">
<table id="allTable">
    <thead><tr>{th}</tr></thead>
    <tbody>{_strat_table(sorted(results, key=lambda r: r.sharpe, reverse=True), 'sharpe')}</tbody>
</table>

<script>
function showTab(evt, tabId) {{
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    evt.target.classList.add('active');
}}

// Filter table by name
document.getElementById('filterInput').addEventListener('input', function() {{
    const val = this.value.toLowerCase();
    document.querySelectorAll('#allTable tbody tr').forEach(row => {{
        const name = row.cells[1].textContent.toLowerCase();
        row.style.display = name.includes(val) ? '' : 'none';
    }});
}});

// Sortable columns
document.querySelectorAll('th').forEach(th => {{
    th.addEventListener('click', function() {{
        const table = this.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.rows);
        const idx = Array.from(this.parentNode.children).indexOf(this);
        const asc = this.dataset.sort !== 'asc';
        this.dataset.sort = asc ? 'asc' : 'desc';
        rows.sort((a, b) => {{
            let va = a.cells[idx].textContent.replace(/[%,d—]/g, '').trim();
            let vb = b.cells[idx].textContent.replace(/[%,d—]/g, '').trim();
            const na = parseFloat(va), nb = parseFloat(vb);
            if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
            return asc ? va.localeCompare(vb) : vb.localeCompare(va);
        }});
        rows.forEach(r => tbody.appendChild(r));
    }});
}});
</script>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)

    return out_path
