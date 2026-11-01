"""Builds the static report (site/index.html) and the in-browser targeting simulator for GitHub Pages."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from .config import OUTCOMES
from .uplift import PRETTY

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
REPO_URL = "https://github.com/dvskaff/email-campaign-uplift"
ARM_COLOR = {"Men's email": BLUE, "Women's email": ORANGE}
RAW_ARM = {"No E-Mail": "No email", "Mens E-Mail": "Men's email", "Womens E-Mail": "Women's email"}


def _style(
    fig: go.Figure, title: str = "", height: int = 380, y_title: str = "", x_title: str = "", **layout
) -> go.Figure:
    base = dict(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        height=height,
        margin=dict(l=70, r=24, t=80 if title else 36, b=50),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK2),
        hovermode="closest",
        hoverlabel=dict(font=dict(family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**{**base, **layout})
    fig.update_xaxes(
        showgrid=False,
        linecolor=AXIS,
        tickfont=dict(color=MUTED),
        zeroline=False,
        title=dict(text=x_title, font=dict(color=MUTED)),
    )
    fig.update_yaxes(
        gridcolor=GRID,
        linecolor=AXIS,
        tickfont=dict(color=MUTED),
        zeroline=False,
        title=dict(text=y_title, font=dict(color=MUTED)),
    )
    return fig


def _div(fig: go.Figure, div_id: str | None = None) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


# --------------------------------------------------------------------------- figures
def fig_balance(balance: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_vrect(x0=-0.1, x1=0.1, fillcolor="#e8f3ec", opacity=0.6, line_width=0, layer="below")
    for arm, color in ARM_COLOR.items():
        b = balance[balance["arm"] == arm]
        fig.add_trace(
            go.Scatter(
                x=b["smd"],
                y=b["covariate"].map({**PRETTY, "history": "Prior-year spend"}),
                mode="markers",
                name=arm,
                marker=dict(color=color, size=10, line=dict(color=SURFACE, width=2)),
                hovertemplate="%{y}: SMD %{x:+.3f}<extra>" + arm + "</extra>",
            )
        )
    _style(fig, "Covariate balance vs. no-email group", height=320, x_title="Standardized mean difference")
    fig.update_xaxes(range=[-0.12, 0.12], zeroline=True, zerolinecolor=AXIS)
    fig.update_yaxes(showgrid=False)
    return fig


def fig_effects(effects: pd.DataFrame) -> go.Figure:
    """Relative lift with 95% CI for every email x outcome."""
    fig = go.Figure()
    order = list(OUTCOMES)
    for arm, color in ARM_COLOR.items():
        e = effects[effects["arm"] == arm].set_index("outcome").loc[order]
        rel = e["relative_lift"] * 100
        lo, hi = e["ci_low"] / e["control_mean"] * 100, e["ci_high"] / e["control_mean"] * 100
        fig.add_trace(
            go.Scatter(
                x=rel,
                y=[OUTCOMES[o] for o in order],
                mode="markers",
                name=arm,
                marker=dict(color=color, size=11, line=dict(color=SURFACE, width=2)),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=hi - rel,
                    arrayminus=rel - lo,
                    color=color,
                    thickness=2,
                    width=0,
                ),
                hovertemplate="%{y}: %{x:+.0f}%<extra>" + arm + "</extra>",
            )
        )
    _style(
        fig,
        "Lift vs. no email (95% confidence interval)",
        height=340,
        x_title="Relative lift, %",
        scattermode="group",
        scattergap=0.5,
    )
    fig.update_xaxes(zeroline=True, zerolinecolor=AXIS, ticksuffix="%")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def fig_mde(mde: dict[str, pd.DataFrame], n_per_arm: int) -> go.Figure:
    fig = go.Figure()
    for (outcome, curve), color in zip(mde.items(), (BLUE, ORANGE), strict=True):
        fig.add_trace(
            go.Scatter(
                x=curve["n_per_arm"],
                y=curve["mde_relative"] * 100,
                name=OUTCOMES[outcome],
                line=dict(color=color, width=2),
                hovertemplate="%{x:,.0f} per group → %{y:.1f}%<extra>" + OUTCOMES[outcome] + "</extra>",
            )
        )
    fig.add_vline(
        x=n_per_arm,
        line=dict(color=MUTED, width=1, dash="dash"),
        annotation_text=f"this test ({n_per_arm:,} per group)",
        annotation_position="top right",
        annotation_font=dict(color=MUTED, size=11),
    )
    _style(
        fig,
        "Smallest detectable lift (80% power, α = 5%)",
        height=360,
        x_title="Customers per group",
        y_title="Minimum detectable lift, % of control",
    )
    fig.update_xaxes(type="log", showgrid=True, gridcolor=GRID)
    return fig


def fig_segments(segment_lift: pd.DataFrame) -> go.Figure:
    seg = segment_lift.copy()
    seg["label"] = seg["dimension"] + ": " + seg["level"].str.replace(r"^\d\) ", "", regex=True)
    labels = list(dict.fromkeys(seg["label"]))
    fig = go.Figure()
    for raw, arm in (("Mens E-Mail", "Men's email"), ("Womens E-Mail", "Women's email")):
        s = seg[seg["arm"] == raw].set_index("label").loc[labels]
        fig.add_trace(
            go.Bar(
                y=labels,
                x=s["visit_lift"] * 100,
                orientation="h",
                name=arm,
                marker=dict(color=ARM_COLOR[arm], cornerradius=3),
                customdata=np.c_[s["customers_in_arm"], s["customers_in_control"]],
                hovertemplate="%{y}<br>%{x:+.1f} pts visit lift<br>%{customdata[0]:,} emailed vs "
                "%{customdata[1]:,} control<extra>" + arm + "</extra>",
            )
        )
    _style(
        fig,
        "Visit-rate lift by customer segment",
        height=max(420, 26 * len(labels) + 120),
        x_title="Lift in visit rate, percentage points",
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, ticksuffix=" pts")
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(size=11))
    return fig


def _usd(v: float) -> str:
    return f"{'+' if v >= 0 else '−'}${abs(v):.2f}"


def fig_qini(curves: dict, arm: int, models: list[str], title: str) -> go.Figure:
    colors = [ORANGE, BLUE, AQUA, YELLOW]
    fig = go.Figure()
    for name, color in zip(models, colors, strict=False):
        c = curves[(arm, "spend", name)]
        fig.add_trace(
            go.Scatter(
                x=c["fraction"] * 100,
                y=c["qini"],
                name=name,
                line=dict(color=color, width=2),
                hovertemplate="Top %{x:.0f}% → $%{y:.3f} per customer<extra>" + name + "</extra>",
            )
        )
    c = curves[(arm, "spend", "Random")]
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, c["qini"].iloc[-1]],
            name="Random targeting",
            mode="lines",
            line=dict(color=MUTED, width=2, dash="dash"),
            hoverinfo="skip",
        )
    )
    _style(
        fig,
        title,
        height=440,
        x_title="% of customers targeted (highest predicted uplift first)",
        y_title="Cumulative extra spend, $ per customer",
        margin=dict(l=70, r=24, t=50, b=130),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(ticksuffix="%")
    return fig


def fig_policies(policies: pd.DataFrame) -> go.Figure:
    p = policies.iloc[::-1]
    colors = [
        ORANGE if n.startswith("Targeted") else (MUTED if n == "Email no one" else BLUE) for n in p["policy"]
    ]
    fig = go.Figure(
        go.Bar(
            x=p["incremental_profit"],
            y=p["policy"],
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            error_x=dict(
                type="data",
                symmetric=False,
                array=p["ci_high"] - p["incremental_profit"],
                arrayminus=p["incremental_profit"] - p["ci_low"],
                color=INK2,
                thickness=1.5,
                width=4,
            ),
            hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
        )
    )
    _style(
        fig,
        "Extra profit per 100k customers (90% CI)",
        height=380,
        showlegend=False,
        x_title="$ vs. emailing no one",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=True,
        zerolinecolor=AXIS,
        tickprefix="$",
        separatethousands=True,
    )
    fig.update_yaxes(showgrid=False)
    return fig


def fig_shap(shap_importance: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for arm, (name, color) in zip((1, 2), ARM_COLOR.items(), strict=True):
        s = shap_importance[shap_importance["arm"] == arm]
        fig.add_trace(
            go.Bar(
                y=s["feature"].astype(str).map(PRETTY),
                x=s["mean_abs_shap"],
                orientation="h",
                name=name,
                marker=dict(color=color, cornerradius=3),
                hovertemplate="%{y}: ±$%{x:.3f}<extra>" + name + "</extra>",
            )
        )
    _style(
        fig,
        "What drives predicted uplift (mean |SHAP|)",
        height=420,
        barmode="group",
        x_title="Average effect on predicted extra spend, $ per customer",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickprefix="$")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


# --------------------------------------------------------------------------- tables
def _effects_table(effects: pd.DataFrame) -> str:
    def fmt(outcome, v):
        return _usd(v) if outcome == "spend" else f"{v * 100:+.2f} pts"

    rows = []
    for r in effects.itertuples(index=False):
        p = "&lt; 0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"
        base = f"${r.control_mean:.2f}" if r.outcome == "spend" else f"{r.control_mean:.2%}"
        boot = (
            (
                f"<br><span class='muted'>bootstrap {fmt(r.outcome, r.boot_ci_low)} to "
                f"{fmt(r.outcome, r.boot_ci_high)}</span>"
            )
            if r.outcome == "spend"
            else ""
        )
        rows.append(
            f"<tr><td>{r.arm}</td><td>{OUTCOMES[r.outcome]}</td><td>{base}</td>"
            f"<td>{fmt(r.outcome, r.lift)}<br><span class='muted'>{fmt(r.outcome, r.ci_low)} to "
            f"{fmt(r.outcome, r.ci_high)}</span>{boot}</td><td>{r.relative_lift:+.0%}</td><td>{p}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Email</th><th>Outcome</th><th>No-email baseline</th><th>Lift (95% CI)</th>"
        "<th>Relative</th><th>Holm p</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _variance_table(effects: pd.DataFrame) -> str:
    rows = []
    for r in effects.itertuples(index=False):
        narrower = 1 - np.sqrt(max(1 - r.reg_var_reduction, 0.0))
        rows.append(
            f"<tr><td>{r.arm}</td><td>{OUTCOMES[r.outcome]}</td>"
            f"<td>{r.cuped_var_reduction:.1%}</td><td>{r.reg_var_reduction:.1%}</td><td>{narrower:.1%}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Email</th><th>Outcome</th><th>CUPED (prior-year spend)</th>"
        "<th>Regression (all covariates)</th><th>CI narrower by</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _qini_table(qini: pd.DataFrame, champion: str) -> str:
    q = qini[qini["outcome"] == "spend"].pivot_table(index="model", columns="arm", values="qini")
    q["mean"] = q.mean(axis=1)
    q = q.sort_values("mean", ascending=False)
    rows = "".join(
        f"<tr class='{'champ' if m == champion else ''}'><td>{html.escape(m)}"
        f"{' <span class=badge>best</span>' if m == champion else ''}</td>"
        f"<td>{r[1]:.4f}</td><td>{r[2]:.4f}</td></tr>"
        for m, r in q.iterrows()
    )
    return (
        "<table><thead><tr><th>Ranking</th><th>Men's email</th><th>Women's email</th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def _arm_table(arm_summary: pd.DataFrame) -> str:
    rows = "".join(
        f"<tr><td>{RAW_ARM.get(r.arm, r.arm)}</td><td>{r.customers:,}</td><td>{r.visit_rate:.2%}</td><td>{r.conversion_rate:.2%}</td>"
        f"<td>${r.spend_per_customer:.2f}</td><td>${r.avg_prior_year_spend:.0f}</td>"
        "</tr>"
        for r in arm_summary.itertuples(index=False)
    )
    return (
        "<table><thead><tr><th>Group</th><th>Customers</th><th>Visited</th><th>Bought</th><th>Spend per "
        "customer</th><th>Prior-year spend</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )


def _card(label: str, value: str, note: str = "") -> str:
    return (
        f"<div class='card'><div class='label'>{label}</div><div class='value'>{value}</div>"
        f"<div class='note'>{note}</div></div>"
    )


# --------------------------------------------------------------------------- simulator
SIMULATOR_JS = r"""
(async function () {
  const fmt = v => (v < 0 ? "−$" : "$") + Math.abs(Math.round(v)).toLocaleString("en-US");
  const el = id => document.getElementById(id);
  let d;
  try { d = await (await fetch("simulator-data.json")).json(); }
  catch (e) { el("sim-status").textContent = "Simulator data could not be loaded."; return; }
  const n = d.T.length;
  // Rank customers once by predicted extra spend from their better email.
  const best = new Int8Array(n), score = new Float64Array(n);
  for (let i = 0; i < n; i++) { best[i] = d.s1[i] >= d.s2[i] ? 1 : 2; score[i] = Math.max(d.s1[i], d.s2[i]); }
  const order = Array.from({length: n}, (_, i) => i).sort((a, b) => score[b] - score[a]);

  // Inverse-propensity estimate of incremental profit (vs emailing no one) for emailing the top k customers.
  function curve(margin, cost) {
    const gain = new Float64Array(n + 1);
    for (let r = 0; r < n; r++) {
      const i = order[r], t = d.T[i];
      const w = (margin * d.Y[i] - cost * (t > 0 ? 1 : 0)) / d.p[t];
      gain[r + 1] = gain[r] + ((t === best[i] ? 1 : 0) - (t === 0 ? 1 : 0)) * w;
    }
    return gain.map(g => g / n * d.per);
  }
  function blast(margin, cost) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const t = d.T[i], w = (margin * d.Y[i] - cost * (t > 0 ? 1 : 0)) / d.p[t];
      s += ((t === 1 ? 1 : 0) - (t === 0 ? 1 : 0)) * w;
    }
    return s / n * d.per;
  }
  function update() {
    const margin = +el("sim-margin").value / 100, cost = +el("sim-cost").value / 100, share = +el("sim-share").value / 100;
    el("sim-margin-out").textContent = Math.round(margin * 100) + "%";
    el("sim-cost-out").textContent = "$" + cost.toFixed(2);
    el("sim-share-out").textContent = Math.round(share * 100) + "%";
    const g = curve(margin, cost), all = blast(margin, cost);
    const k = Math.round(share * n), chosen = g[k];
    let bestK = 0; for (let j = 0; j <= n; j += Math.ceil(n / 200)) if (g[j] > g[bestK]) bestK = j;
    el("sim-chosen").textContent = fmt(chosen);
    el("sim-blast").textContent = fmt(all);
    el("sim-best").textContent = fmt(g[bestK]) + " at " + Math.round(bestK / n * 100) + "% emailed";
    const narrow = window.innerWidth < 640, xs = [], ys = [];
    for (let s = 0; s <= 100; s += 2) { xs.push(s); ys.push(g[Math.round(s / 100 * n)]); }
    Plotly.react("sim-chart", [
      {x: xs, y: ys, mode: "lines", name: "Targeted by the model", line: {color: "%ORANGE%", width: 2},
       hovertemplate: "Email top %{x}% → %{y:$,.0f}<extra></extra>"},
      {x: [0, 100], y: [all, all], mode: "lines", name: "Men's email to everyone",
       line: {color: "%BLUE%", width: 2, dash: "dash"}, hoverinfo: "skip"},
      {x: [share * 100], y: [chosen], mode: "markers", name: "Your choice",
       marker: {color: "%ORANGE%", size: 12, line: {color: "%SURFACE%", width: 2}}, hoverinfo: "skip"}
    ], {height: narrow ? 420 : 340, margin: {l: 80, r: 20, t: 30, b: narrow ? 150 : 50}, paper_bgcolor: "%SURFACE%", plot_bgcolor: "%SURFACE%",
        font: {family: %FONTJS%, size: 12, color: "%INK2%"}, showlegend: true,
        legend: narrow ? {orientation: "h", yanchor: "top", y: -0.3, xanchor: "left", x: 0}
                      : {orientation: "h", yanchor: "bottom", y: 1.0, xanchor: "right", x: 1},
        xaxis: {title: {text: "% of customers emailed (highest predicted uplift first)"}, ticksuffix: "%",
                linecolor: "%AXIS%", tickfont: {color: "%MUTED%"}},
        yaxis: {title: {text: "Incremental profit per 100k customers"}, tickprefix: "$", separatethousands: true,
                gridcolor: "%GRID%", tickfont: {color: "%MUTED%"}, zeroline: true, zerolinecolor: "%AXIS%"}},
      {displaylogo: false, responsive: true});
  }
  ["sim-margin", "sim-cost", "sim-share"].forEach(id => el(id).addEventListener("input", update));
  el("sim-status").textContent = "";
  update();
})();
"""


LAYOUT_JS = r"""
  window.addEventListener("load", () => document.querySelectorAll(".plotly-graph-div").forEach(d => {
    // On phones, move legends under the chart so they never collide with the title.
    if (window.innerWidth < 640 && d.layout && d.layout.showlegend !== false && d.data.length > 1) {
      Plotly.relayout(d, {"legend.y": -0.28, "legend.yanchor": "top", "legend.x": 0, "legend.xanchor": "left",
                          "margin.t": 50, "margin.b": Math.max((d.layout.margin || {}).b || 0, 140)});
    }
    Plotly.Plots.resize(d);
  }));
"""


def _simulator_js() -> str:
    js = SIMULATOR_JS
    for key, val in {
        "ORANGE": ORANGE,
        "BLUE": BLUE,
        "SURFACE": SURFACE,
        "INK2": INK2,
        "AXIS": AXIS,
        "MUTED": MUTED,
        "GRID": GRID,
    }.items():
        js = js.replace(f"%{key}%", val)
    return js.replace("%FONTJS%", json.dumps(FONT))


def _budget_peak(sim: dict, margin: float, cost: float) -> tuple[float, float, float]:
    """Same calculation as the in-browser simulator: best share to email, its profit, and the men's blast."""
    T, Y = np.asarray(sim["T"]), np.asarray(sim["Y"], float)
    s = np.c_[sim["s1"], sim["s2"]]
    w = (margin * Y - cost * (T > 0)) / np.asarray(sim["p"])[T]
    best = s.argmax(axis=1) + 1
    order = np.argsort(-s.max(axis=1), kind="stable")
    gain = np.r_[0.0, np.cumsum((((T == best).astype(float) - (T == 0)) * w)[order])] / len(T) * sim["per"]
    blast = float((((T == 1).astype(float) - (T == 0)) * w).mean() * sim["per"])
    k = int(gain.argmax())
    return k / len(T), float(gain[k]), blast


# --------------------------------------------------------------------------- page
def build_site(
    site_dir: Path,
    *,
    results,
    arm_summary,
    segment_lift,
    balance,
    effects,
    mde,
    qini,
    curves,
    policies,
    budget_curve,
    shap_importance,
    simulator,
) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "simulator-data.json").write_text(json.dumps(simulator, separators=(",", ":")))
    e = effects.set_index(["arm", "outcome"])
    men_v, men_s, wom_v = (
        e.loc[("Men's email", "visit")],
        e.loc[("Men's email", "spend")],
        e.loc[("Women's email", "visit")],
    )
    champion = results["champion"]
    best_policy = policies.iloc[0]
    n_per_arm = int(np.mean(list(results["arm_counts"].values())))
    qini_models = [champion] + [
        m for m in ("Causal forest (spend)", "Rule: matches past purchases") if m != champion
    ]
    blast = policies.set_index("policy").loc["Men's email to everyone"]
    best_budget = budget_curve.loc[budget_curve["incremental_profit"].idxmax()]
    share_hi, profit_hi, blast_hi = _budget_peak(simulator, results["margin"], 0.30)
    if best_budget["budget"] >= 0.95:
        peak_text = (
            f"At ${results['email_cost']:.2f} an email almost everyone is worth emailing, so the gain comes from "
            "sending each customer the <i>right</i> email. Targeting matters more as emails get more expensive: "
            f"at $0.30 per email the best plan emails only the top {share_hi:.0%} of customers and earns "
            f"${profit_hi:,.0f} per 100,000 vs. ${blast_hi:,.0f} for the men's email to everyone."
        )
    else:
        peak_text = (
            f"At these settings the profit curve peaks when emailing the top {best_budget['budget']:.0%} of "
            "customers; emailing the rest costs more than it brings in."
        )
    sql_arm = html.escape((Path(__file__).resolve().parents[2] / "sql" / "arm_summary.sql").read_text())

    cards = "".join(
        [
            _card(
                "Customers",
                f"{results['n_customers']:,}",
                f"3 random groups · sample-ratio check p = {results['srm_p']:.2f} ✓",
            ),
            _card(
                "Men's email · visits",
                f"{men_v['lift'] * 100:+.1f} pts",
                f"{men_v['relative_lift']:+.0%} vs. no email ({men_v['control_mean']:.1%} → {men_v['treated_mean']:.1%})",
            ),
            _card(
                "Men's email · spend",
                _usd(men_s["lift"]),
                f"per customer, {men_s['relative_lift']:+.0%} vs. no email",
            ),
            _card(
                "Women's email · visits",
                f"{wom_v['lift'] * 100:+.1f} pts",
                f"{wom_v['relative_lift']:+.0%} vs. no email",
            ),
            _card(
                "Best targeting",
                f"${best_policy['incremental_profit']:,.0f}",
                f"per 100k customers: {html.escape(best_policy['policy'].replace('Targeted: ', 'targeted by '))}",
            ),
        ]
    )

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Email Experiment & Uplift</title>
<meta name="description" content="A/B test analysis, CUPED and uplift modeling on a 64,000-customer randomized email experiment, with a targeting simulator.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>✉️</text></svg>">
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>
<style>
  :root {{ --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --grid:{GRID}; --surface:{SURFACE}; --page:#f9f9f7;
          --accent:{ORANGE}; --blue:{BLUE}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--page); color:var(--ink); font-family:{FONT}; line-height:1.55; }}
  header {{ background:#104281; color:#fff; padding:40px 16px 32px; }}
  header .wrap, main {{ max-width:1120px; margin:0 auto; }}
  header h1 {{ margin:0 0 6px; font-size:clamp(24px,4vw,34px); letter-spacing:-0.01em; }}
  header p {{ margin:0; color:#cde2fb; max-width:780px; }}
  header a {{ color:#fff; }}
  main {{ padding:24px 16px 64px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-bottom:28px; }}
  .card {{ background:var(--surface); border:1px solid rgba(11,11,11,.10); border-radius:12px; padding:14px 16px; }}
  .card .label {{ font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
  .card .value {{ font-size:24px; font-weight:650; margin:4px 0 2px; }}
  .card .note {{ font-size:12.5px; color:var(--ink2); }}
  section {{ background:var(--surface); border:1px solid rgba(11,11,11,.10); border-radius:12px; padding:20px 20px 8px;
             margin-bottom:20px; }}
  section h2 {{ margin:0 0 4px; font-size:19px; }}
  section > p, .note-p {{ margin:4px 0 12px; color:var(--ink2); max-width:880px; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr)); gap:8px 20px; }}
  .grid2 > * {{ min-width:0; }}
  h3 {{ font-size:15px; margin:14px 0 4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; font-variant-numeric:tabular-nums; margin:8px 0 16px; }}
  th, td {{ text-align:right; padding:8px 10px; border-bottom:1px solid var(--grid); vertical-align:top; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  tr.champ td {{ font-weight:650; }}
  .badge {{ background:var(--accent); color:#fff; border-radius:999px; padding:1px 8px; font-size:11px; margin-left:6px; }}
  .muted {{ color:var(--muted); font-size:12.5px; }}
  .tablewrap {{ overflow-x:auto; }}
  code, pre {{ background:#f0efec; border-radius:6px; font-size:13px; }}
  code {{ padding:1px 5px; }}
  pre {{ padding:12px 14px; overflow-x:auto; line-height:1.45; }}
  details summary {{ cursor:pointer; color:var(--blue); margin:4px 0 10px; }}
  .sim {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr)); gap:16px; margin:8px 0 6px; }}
  .sim label {{ display:block; font-size:13px; color:var(--ink2); }}
  .sim label b {{ float:right; color:var(--ink); }}
  .sim input[type=range] {{ width:100%; accent-color:var(--accent); }}
  .simout {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:12px; margin:6px 0 4px; }}
  .simout div {{ background:#f6f5f2; border-radius:10px; padding:10px 12px; }}
  .simout span {{ display:block; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
  .simout b {{ font-size:20px; }}
  ul.method li {{ margin-bottom:6px; color:var(--ink2); }}
  @media (max-width:640px) {{
    section {{ padding:16px 12px 6px; }}
    table {{ font-size:12.5px; }}
    th, td {{ padding:6px 7px; white-space:nowrap; }}
  }}
  footer {{ text-align:center; color:var(--muted); font-size:13px; padding:0 16px 40px; }}
</style></head>
<body>
<header><div class="wrap">
  <h1>Email Campaign Experiment: A/B Testing &amp; Uplift Modeling</h1>
  <p>{results["n_customers"]:,} customers were randomly sent a men's merchandise email, a women's merchandise email, or
  nothing, then tracked for two weeks. Did the emails work, for whom, and who should get one next time?
  <a href="{REPO_URL}">Code on GitHub</a></p>
</div></header>
<main>
<div class="cards">{cards}</div>

<section><h2>1. Is the experiment trustworthy?</h2>
<p>Before reading any results: the three groups should be the same size and look alike on everything measured
<i>before</i> the emails went out. A sample-ratio test gives p = {results["srm_p"]:.2f} (no sign of broken
randomization), and every pre-experiment characteristic is within ±{results["max_abs_smd"]:.3f} standard deviations
across groups, well inside the usual ±0.1 limit.</p>
<div class="grid2">{_div(fig_balance(balance))}
<div><h3>Group summary (DuckDB SQL)</h3><div class="tablewrap">{_arm_table(arm_summary)}</div>
<details><summary>Show the SQL</summary><pre>{sql_arm}</pre></details></div></div></section>

<section><h2>2. What did each email do?</h2>
<p>Difference in means with 95% confidence intervals. With six tests (2 emails × 3 outcomes), p-values are
Holm-adjusted so the chance of any false positive stays under 5%. Spend is heavily skewed (about 99% of
customers spend $0), so its interval is also checked with a bootstrap, which makes no normality assumption.</p>
{_div(fig_effects(effects))}<div class="tablewrap">{_effects_table(effects)}</div></section>

<section><h2>3. Can we measure it more precisely? (CUPED)</h2>
<p>CUPED reduces noise by subtracting the part of each customer's outcome predicted by data from <i>before</i> the
test, here their prior-year spend. Regression adjustment extends the same idea to every pre-experiment
covariate. Both give the same lift estimates (randomization guarantees that) but with tighter intervals.
<b>Honest result:</b> prior-year spend barely predicts two-week behaviour, so variance falls by under 1% with CUPED
and up to {effects["reg_var_reduction"].max():.0%} with all covariates. Variance reduction is only as good as the
pre-period signal.</p>
<div class="grid2"><div class="tablewrap"><h3>Variance reduction</h3>{_variance_table(effects)}</div>
{_div(fig_mde(mde, n_per_arm))}</div></section>

<section><h2>4. Who responds? (DuckDB SQL)</h2>
<p>Lift in visit rate within each segment. Past buyers of both categories respond most to the men's
email; men's-only buyers barely react to the women's email, while women's-only buyers respond to both about
equally. Segment cuts are descriptive: with many segments, some differences
are noise, which is why the next section uses models evaluated on customers they never saw.</p>
{_div(fig_segments(segment_lift))}</section>

<section><h2>5. Ranking customers by predicted uplift</h2>
<p>Uplift models predict how much <i>each</i> customer's spend would rise if emailed. Models were trained with
{results["crossfit_folds"]}-fold cross-fitting, so every customer was scored by a model that never saw them. A Qini
curve shows the extra spend captured by emailing customers in the model's order; the further above the random
line, the better the ranking. Purchases are rare (about 1%), so models trained directly on spend are noisy; the
causal forest trained on visits (converted to dollars at ${results["spend_per_visit"][0]:.2f} per extra visit for
the men's email) ranks spend best.</p>
<div class="grid2">{_div(fig_qini(curves, 1, qini_models, "Men's email: Qini curve (spend)"))}
{_div(fig_qini(curves, 2, qini_models, "Women's email: Qini curve (spend)"))}</div>
<h3>Qini coefficient on spend (higher is better; random ≈ 0)</h3><div class="tablewrap">{_qini_table(qini, champion)}</div>
</section>

<section><h2>6. Who should get an email?</h2>
<p>Each policy is valued on the actual experiment using inverse-propensity weighting: customers whose random
assignment matches what the policy would have sent are reweighted to stand in for everyone. Assumes a
{results["margin"]:.0%} margin on extra spend and ${results["email_cost"]:.2f} per email. The best policy here
earns <b>${best_policy["incremental_profit"]:,.0f}</b> per 100,000 customers vs.
${blast["incremental_profit"]:,.0f} for sending the men's email to everyone. {peak_text}</p>
{_div(fig_policies(policies))}
<h3>Targeting simulator</h3>
<p class="note-p">Change the economics and the share of customers you email. Customers are emailed in order of the
champion model's ({html.escape(champion.lower())}) predicted extra spend, each with whichever email suits them better. Profit is estimated
live in your browser from the {results["n_customers"]:,} real customers.</p>
<div class="sim">
  <label>Gross margin on extra spend <b id="sim-margin-out"></b><input id="sim-margin" type="range" min="10" max="80" step="5" value="{int(results["margin"] * 100)}"></label>
  <label>Cost per email <b id="sim-cost-out"></b><input id="sim-cost" type="range" min="0" max="100" step="1" value="{int(round(results["email_cost"] * 100))}"></label>
  <label>Share of customers emailed <b id="sim-share-out"></b><input id="sim-share" type="range" min="0" max="100" step="1" value="{int(round(best_budget["budget"] * 100))}"></label>
</div>
<div class="simout"><div><span>Your plan</span><b id="sim-chosen">…</b></div>
<div><span>Men's email to everyone</span><b id="sim-blast">…</b></div>
<div><span>Best share at these settings</span><b id="sim-best">…</b></div></div>
<p class="muted" id="sim-status">Loading simulator…</p>
<div id="sim-chart"></div></section>

<section><h2>7. What drives predicted uplift? (SHAP)</h2>
<p>SHAP values for a LightGBM surrogate trained to reproduce the champion model's predictions ({html.escape(champion.lower())})
(R² = {results["surrogate_r2"][1]:.2f} for the men's email, {results["surrogate_r2"][2]:.2f} for the women's), in
dollars of predicted extra spend.</p>
{_div(fig_shap(shap_importance))}</section>

<section><h2>How it was built</h2><ul class="method">
<li><b>Data:</b> Kevin Hillstrom's MineThatData E-Mail Analytics Challenge (2008): 64,000 customers who purchased
in the prior year, randomized into three equal groups. Loaded and summarized with DuckDB SQL.</li>
<li><b>A/B analysis:</b> sample-ratio and covariate-balance checks; Welch confidence intervals; Holm correction;
bootstrap intervals for spend; CUPED and regression adjustment (statsmodels); minimum detectable effects.</li>
<li><b>Uplift:</b> EconML causal forests (double machine learning with LightGBM nuisance models) and X-learners, on
visits and on spend; {results["crossfit_folds"]}-fold cross-fitting; Qini curves vs. a business rule and random.</li>
<li><b>Targeting:</b> per-customer best email by expected profit, valued with inverse-propensity weighting and
bootstrap intervals.</li>
<li><b>Explainability:</b> TreeSHAP on a LightGBM surrogate of the causal forest.</li>
<li><b>Automation:</b> pytest + ruff on every push; GitHub Actions reruns the analysis and redeploys this page.</li>
<li><b>Limits:</b> one two-week campaign from 2008; spend is noisy, so policy values have wide intervals; costs and
margins are assumptions (adjust them above).</li></ul></section>
</main>
<script>{_simulator_js()}</script>
<script>{LAYOUT_JS}</script>
<footer>Data: MineThatData E-Mail Analytics Challenge (Kevin Hillstrom, 2008) · <a href="{REPO_URL}">Source code</a></footer>
</body></html>"""
    out = site_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    (site_dir / ".nojekyll").touch()
    return out
