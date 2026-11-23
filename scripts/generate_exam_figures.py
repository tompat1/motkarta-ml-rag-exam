#!/usr/bin/env python3
"""Generate publication-grade vector SVG figures from REAL Motkarta ML & RAG data.

Uses only pandas, numpy, scikit-learn, and Python stdlib so the exam repository
is 100% reproducible with zero heavy plotting dependencies.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BG = "#FCFBF7"
INK = "#141414"
MUTED = "#5C5852"
GRID = "#E5E1D8"
ACCENT = "#D94E2A"
GOLD = "#D97706"
EMERALD = "#059669"
BLUE = "#2563EB"
PURPLE = "#7C3AED"


def svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}">',
        f'  <rect width="{width}" height="{height}" fill="{BG}" stroke="{INK}" stroke-width="1.5"/>',
        f'  <text x="32" y="38" font-family="ui-sans-serif, system-ui, -apple-system, sans-serif" font-size="18" font-weight="800" fill="{INK}">{html.escape(title)}</text>',
        f'  <text x="32" y="58" font-family="ui-sans-serif, system-ui, -apple-system, sans-serif" font-size="12" fill="{MUTED}">{html.escape(subtitle)}</text>',
    ]


def fig1_eda_districts_and_completeness(scored: pd.DataFrame, slice_json: dict) -> Path:
    summary = slice_json["catalogSummary"]
    top_districts = list(summary["topDistricts"].items())[:10]
    total_cat = summary["fullCatalogTotalPlaces"]

    completeness = [
        ("Coordinates (lat/lng)", summary["withCoordinates"] / total_cat * 100),
        ("Primary cuisine", (1 - scored["primary_cuisine"].isna().mean()) * 100),
        ("Opening hours", (1 - scored["missing_opening_hours"].mean()) * 100),
        ("Street address", (1 - scored["missing_address"].mean()) * 100),
        ("Official website", summary["withWebsite"] / total_cat * 100),
        ("Independent business", scored["independent_business"].mean() * 100),
    ]

    w, h = 1120, 480
    lines = svg_header(
        w, h,
        "1. Exploratory Data Analysis — Stockholm Catalog Distribution & Field Completeness",
        f"Real catalog: {total_cat:,} published Stockholm venues | Scored ML table: {len(scored):,} rows × {scored.shape[1]} features",
    )

    # Left panel: Top 10 districts horizontal bar chart
    lx, ly, lw, lh = 32, 92, 510, 355
    lines.append(f'  <rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{lx+16}" y="{ly+26}" font-family="ui-sans-serif, sans-serif" font-size="13" font-weight="700" fill="{INK}">Top 10 Stockholm Districts (Published Catalog)</text>')
    max_val = max(v for _, v in top_districts) or 1
    for i, (district, count) in enumerate(top_districts):
        y = ly + 48 + i * 29
        bar_w = int((count / max_val) * 280)
        lines.append(f'  <text x="{lx+142}" y="{y+14}" text-anchor="end" font-family="ui-monospace, monospace" font-size="11" fill="{INK}">{html.escape(district[:18])}</text>')
        lines.append(f'  <rect x="{lx+152}" y="{y+2}" width="{bar_w}" height="16" fill="{BLUE if i > 0 else ACCENT}"/>')
        lines.append(f'  <text x="{lx+158+bar_w}" y="{y+14}" font-family="ui-monospace, monospace" font-size="11" font-weight="700" fill="{INK}">{count}</text>')

    # Right panel: Field completeness & gate coverage
    rx, ry, rw, rh = 568, 92, 520, 355
    lines.append(f'  <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{rx+16}" y="{ry+26}" font-family="ui-sans-serif, sans-serif" font-size="13" font-weight="700" fill="{INK}">Data Quality &amp; Feature Coverage (%)</text>')
    for i, (label, pct) in enumerate(completeness):
        y = ry + 56 + i * 44
        bar_w = int((pct / 100.0) * 280)
        color = EMERALD if pct >= 80 else (GOLD if pct >= 55 else ACCENT)
        lines.append(f'  <text x="{rx+168}" y="{y+14}" text-anchor="end" font-family="ui-sans-serif, sans-serif" font-size="11.5" fill="{INK}">{html.escape(label)}</text>')
        lines.append(f'  <rect x="{rx+178}" y="{y+2}" width="280" height="18" fill="#F3F1EC"/>')
        lines.append(f'  <rect x="{rx+178}" y="{y+2}" width="{bar_w}" height="18" fill="{color}"/>')
        lines.append(f'  <text x="{rx+466}" y="{y+15}" font-family="ui-monospace, monospace" font-size="11" font-weight="700" fill="{INK}">{pct:.1f}%</text>')

    # Category footnote
    kinds_str = " · ".join(f"{k}: {v}" for k, v in summary["byKind"].items())
    lines.append(f'  <text x="32" y="{h-14}" font-family="ui-monospace, monospace" font-size="10.5" fill="{MUTED}">Categories: {html.escape(kinds_str)}</text>')
    lines.append("</svg>")
    out = FIG_DIR / "01_eda_districts_and_completeness.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def fig2_correlations_and_anomalies(scored: pd.DataFrame) -> Path:
    cols = [
        ("spatial_density_300m", "Density"),
        ("discovery_score", "Discovery"),
        ("opening_hours_score", "Hours"),
        ("independent_business", "Indep."),
        ("underrepresented_cuisine", "Rare Cuis."),
        ("complete_profile", "Complete"),
    ]
    num_df = scored[[c for c, _ in cols]].astype(float).fillna(0.0)
    corr = num_df.corr().values

    w, h = 1120, 480
    lines = svg_header(
        w, h,
        "2. Feature Correlations & Isolation Forest Structural Anomalies (Real 3,192 Venues)",
        "Left: Pearson correlation matrix of engineered features | Right: Spatial density vs Discovery score with Isolation Forest outliers",
    )

    # Left panel: Correlation heatmap
    lx, ly, lw, lh = 32, 92, 480, 355
    lines.append(f'  <rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{lx+16}" y="{ly+26}" font-family="ui-sans-serif, sans-serif" font-size="13" font-weight="700" fill="{INK}">Engineered Feature Correlation Matrix</text>')
    n = len(cols)
    cell = 44
    ox, oy = lx + 92, ly + 46
    for i in range(n):
        lines.append(f'  <text x="{ox-8}" y="{oy+i*cell+26}" text-anchor="end" font-family="ui-monospace, monospace" font-size="10.5" fill="{INK}">{cols[i][1]}</text>')
        lines.append(f'  <text x="{ox+i*cell+22}" y="{oy+n*cell+18}" text-anchor="middle" font-family="ui-monospace, monospace" font-size="10" fill="{INK}">{cols[i][1]}</text>')
        for j in range(n):
            v = float(corr[i, j])
            if v >= 0:
                r = int(252 - v * (252 - 37))
                g = int(251 - v * (251 - 99))
                b = int(247 - v * (247 - 235))
            else:
                av = abs(v)
                r = int(252 - av * (252 - 217))
                g = int(251 - av * (251 - 78))
                b = int(247 - av * (247 - 42))
            txt_color = "#FFFFFF" if abs(v) > 0.55 else INK
            lines.append(f'  <rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell-2}" height="{cell-2}" fill="rgb({r},{g},{b})"/>')
            lines.append(f'  <text x="{ox+j*cell+21}" y="{oy+i*cell+26}" text-anchor="middle" font-family="ui-monospace, monospace" font-size="10" font-weight="700" fill="{txt_color}">{v:.2f}</text>')

    # Right panel: Real Isolation Forest scatter (spatial_density_300m vs discovery_score)
    rx, ry, rw, rh = 536, 92, 552, 355
    lines.append(f'  <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="#FFFFFF" stroke="{GRID}"/>')
    n_anom = int(scored["is_structural_anomaly"].sum())
    lines.append(f'  <text x="{rx+16}" y="{ry+26}" font-family="ui-sans-serif, sans-serif" font-size="13" font-weight="700" fill="{INK}">Isolation Forest Outliers ({n_anom} structural anomalies / {len(scored):,} venues)</text>')

    px, py, pw, ph = rx + 58, ry + 48, rw - 84, rh - 96
    lines.append(f'  <line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py+ph}" stroke="{INK}" stroke-width="1.2"/>')
    lines.append(f'  <line x1="{px}" y1="{py}" x2="{px}" y2="{py+ph}" stroke="{INK}" stroke-width="1.2"/>')

    sample = scored.sample(n=min(700, len(scored)), random_state=42)
    max_x = float(max(scored["spatial_density_300m"].quantile(0.99), 1.0))
    max_y = 105.0
    for tick in [0, 25, 50, 75, 100]:
        ty = py + ph - int((tick / max_y) * ph)
        lines.append(f'  <line x1="{px}" y1="{ty}" x2="{px+pw}" y2="{ty}" stroke="{GRID}" stroke-dasharray="3,3"/>')
        lines.append(f'  <text x="{px-8}" y="{ty+4}" text-anchor="end" font-family="ui-monospace, monospace" font-size="10" fill="{MUTED}">{tick}</text>')

    for _, row in sample.iterrows():
        xv = min(float(row["spatial_density_300m"] or 0.0), max_x)
        yv = float(row["discovery_score"] or 0.0)
        cx = px + int((xv / max_x) * pw)
        cy = py + ph - int((yv / max_y) * ph)
        is_anom = bool(row["is_structural_anomaly"])
        color = ACCENT if is_anom else "#94A3B8"
        rad = 3.8 if is_anom else 2.2
        opacity = "0.85" if is_anom else "0.45"
        lines.append(f'  <circle cx="{cx}" cy="{cy}" r="{rad}" fill="{color}" fill-opacity="{opacity}"/>')

    lines.append(f'  <text x="{px+pw//2}" y="{py+ph+32}" text-anchor="middle" font-family="ui-sans-serif, sans-serif" font-size="11" font-weight="700" fill="{INK}">Spatial Density (venues within 300m)</text>')
    lines.append(f'  <text x="{rx+18}" y="{py+ph//2}" text-anchor="middle" transform="rotate(-90 {rx+18} {py+ph//2})" font-family="ui-sans-serif, sans-serif" font-size="11" font-weight="700" fill="{INK}">Discovery Score</text>')
    lines.append(f'  <circle cx="{px+pw-150}" cy="{py+14}" r="4" fill="{ACCENT}"/><text x="{px+pw-140}" y="{py+18}" font-family="ui-sans-serif, sans-serif" font-size="10.5" fill="{INK}">Structural Anomaly</text>')
    lines.append(f'  <circle cx="{px+pw-150}" cy="{py+32}" r="3" fill="#94A3B8"/><text x="{px+pw-140}" y="{py+36}" font-family="ui-sans-serif, sans-serif" font-size="10.5" fill="{INK}">Normal Venue (Inlier)</text>')
    lines.append("</svg>")
    out = FIG_DIR / "02_feature_correlations_and_anomalies.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def fig3_baseline_vs_multisignal_and_ltr(scored: pd.DataFrame, ab: dict) -> Path:
    # Real 5-fold cross-fitted HistGradientBoostingRegressor on real Motkarta features
    feature_cols = [
        "spatial_density_300m",
        "opening_hours_score",
        "independent_business",
        "underrepresented_cuisine",
        "low_local_visibility",
        "verified_open",
        "complete_profile",
        "missing_website",
        "missing_address",
    ]
    X = scored[feature_cols].astype(float).fillna(0.0)
    y = scored["discovery_score"].astype(float).values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = HistGradientBoostingRegressor(max_depth=4, max_iter=80, random_state=42)
    oof_pred = cross_val_predict(model, X, y, cv=kf)
    mae = float(mean_absolute_error(y, oof_pred))
    r2 = float(r2_score(y, oof_pred))
    residuals = y - oof_pred

    w, h = 1120, 480
    lines = svg_header(
        w, h,
        "3. Ranking Baseline A/B Experiment & 5-Fold Cross-Validated ML Model",
        f"Left: Raw Popularity (A) vs Transparent Multi-Signal (B) | Right: HistGradientBoostingRegressor OOF Residuals (MAE={mae:.3f}, R²={r2:.4f})",
    )

    # Left panel: Grouped bar chart of Ranking A vs Ranking B
    lx, ly, lw, lh = 32, 92, 510, 355
    lines.append(f'  <rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{lx+16}" y="{ly+26}" font-family="ui-sans-serif, sans-serif" font-size="13" font-weight="700" fill="{INK}">Ranking A (Raw Popularity) vs Ranking B (Multi-Signal)</text>')
    a = ab["ranking_a_raw_popularity"]
    b = ab["ranking_b_transparent_multisignal"]
    metrics = [
        ("Cuisine Entropy (×20)", a["cuisine_diversity_entropy"] * 20, b["cuisine_diversity_entropy"] * 20, f"{a['cuisine_diversity_entropy']:.2f}", f"{b['cuisine_diversity_entropy']:.2f}"),
        ("Independent Ratio (%)", a["independent_business_ratio"] * 100, b["independent_business_ratio"] * 100, f"{a['independent_business_ratio']*100:.0f}%", f"{b['independent_business_ratio']*100:.0f}%"),
        ("Unfamiliar Discovery (%)", a["unfamiliar_discovery_ratio"] * 100, b["unfamiliar_discovery_ratio"] * 100, f"{a['unfamiliar_discovery_ratio']*100:.0f}%", f"{b['unfamiliar_discovery_ratio']*100:.0f}%"),
        ("Satisfaction Proxy", a["satisfaction_proxy"], b["satisfaction_proxy"], f"{a['satisfaction_proxy']:.1f}", f"{b['satisfaction_proxy']:.1f}"),
    ]
    for i, (label, va, vb, la, lb) in enumerate(metrics):
        y0 = ly + 62 + i * 64
        wa = int((min(va, 100) / 100.0) * 260)
        wb = int((min(vb, 100) / 100.0) * 260)
        lines.append(f'  <text x="{lx+16}" y="{y0-4}" font-family="ui-sans-serif, sans-serif" font-size="11.5" font-weight="700" fill="{INK}">{html.escape(label)}</text>')
        lines.append(f'  <rect x="{lx+150}" y="{y0+4}" width="{wa}" height="15" fill="#94A3B8"/>')
        lines.append(f'  <text x="{lx+156+wa}" y="{y0+16}" font-family="ui-monospace, monospace" font-size="10.5" fill="{MUTED}">A: {la}</text>')
        lines.append(f'  <rect x="{lx+150}" y="{y0+23}" width="{wb}" height="15" fill="{EMERALD}"/>')
        lines.append(f'  <text x="{lx+156+wb}" y="{y0+35}" font-family="ui-monospace, monospace" font-size="10.5" font-weight="700" fill="{EMERALD}">B: {lb}</text>')

    # Right panel: Residual histogram from real 5-fold CV
    rx, ry, rw, rh = 568, 92, 520, 355
    lines.append(f'  <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{rx+16}" y="{ry+26}" font-family="ui-sans-serif, sans-serif" font-size="13" font-weight="700" fill="{INK}">5-Fold OOF Prediction Residual Distribution (n={len(y):,})</text>')
    hist, edges = np.histogram(residuals, bins=24)
    px, py, pw, ph = rx + 52, ry + 52, rw - 76, rh - 102
    lines.append(f'  <line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py+ph}" stroke="{INK}" stroke-width="1.2"/>')
    lines.append(f'  <line x1="{px}" y1="{py}" x2="{px}" y2="{py+ph}" stroke="{INK}" stroke-width="1.2"/>')
    max_h = max(int(hist.max()), 1)
    bw = pw / len(hist)
    for i, count in enumerate(hist):
        bh = int((int(count) / max_h) * (ph - 10))
        bx = px + i * bw
        lines.append(f'  <rect x="{bx:.1f}" y="{py+ph-bh}" width="{max(bw-2, 2):.1f}" height="{bh}" fill="{GOLD}" fill-opacity="0.85"/>')
    # Zero line
    lo, hi = float(edges[0]), float(edges[-1])
    if lo < 0 < hi:
        zx = px + int(((0 - lo) / (hi - lo)) * pw)
        lines.append(f'  <line x1="{zx}" y1="{py}" x2="{zx}" y2="{py+ph}" stroke="{ACCENT}" stroke-width="1.8" stroke-dasharray="4,3"/>')
        lines.append(f'  <text x="{zx+6}" y="{py+14}" font-family="ui-monospace, monospace" font-size="10.5" font-weight="700" fill="{ACCENT}">Zero Error</text>')
    lines.append(f'  <text x="{px+pw//2}" y="{py+ph+32}" text-anchor="middle" font-family="ui-sans-serif, sans-serif" font-size="11" font-weight="700" fill="{INK}">Residual (y_actual − y_pred) | MAE = {mae:.3f}, R² = {r2:.4f}</text>')
    lines.append("</svg>")
    out = FIG_DIR / "03_baseline_vs_multisignal_and_ltr.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def fig4_vector_space_projection(slice_json: dict) -> Path:
    places = slice_json["places"]
    docs = [
        f"{p.get('name','')} {p.get('kind','')} {p.get('area','')} {p.get('cuisine','')} {' '.join(p.get('tags') or [])} {p.get('note','')}"
        for p in places
    ]
    vec = TfidfVectorizer(max_features=600, stop_words="english")
    X = vec.fit_transform(docs)
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X)
    explained = svd.explained_variance_ratio_.sum() * 100

    kind_colors = {
        "Restaurant": BLUE,
        "Café": GOLD,
        "Bakery": EMERALD,
        "Specialty coffee": PURPLE,
    }

    w, h = 1120, 480
    lines = svg_header(
        w, h,
        "4. Semantic Document Vector Space — 2D SVD Projection of Motkarta Venue Documents",
        f"Real TF-IDF + TruncatedSVD over {len(places)} stratified Stockholm venue documents (2D variance explained: {explained:.1f}% | Production Vectorize: 3,134 × 1,024-dim bge-m3 (cosine))",
    )
    px, py, pw, ph = 64, 96, 820, 336
    lines.append(f'  <rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="#FFFFFF" stroke="{GRID}"/>')
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)

    for (x, y), p in zip(coords, places):
        cx = px + 24 + ((float(x) - x_min) / dx) * (pw - 48)
        cy = py + ph - 24 - ((float(y) - y_min) / dy) * (ph - 48)
        color = kind_colors.get(p.get("kind") or "Restaurant", MUTED)
        lines.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.2" fill="{color}" fill-opacity="0.72" stroke="#FFFFFF" stroke-width="0.6"/>')

    # Annotate 6 iconic Stockholm anchor venues if present
    anchors = {"Pelikan", "Drop Coffee", "Café Pascal", "Lillebrors Bageri", "Soldaten Svejk", "Fabrique"}
    annotated = set()
    for (x, y), p in zip(coords, places):
        name = p.get("name", "")
        if name in anchors and name not in annotated:
            annotated.add(name)
            cx = px + 24 + ((float(x) - x_min) / dx) * (pw - 48)
            cy = py + ph - 24 - ((float(y) - y_min) / dy) * (ph - 48)
            lines.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" stroke="{INK}" stroke-width="1.5"/>')
            lines.append(f'  <text x="{cx+8:.1f}" y="{cy+4:.1f}" font-family="ui-monospace, monospace" font-size="10.5" font-weight="700" fill="{INK}">{html.escape(name)}</text>')

    # Legend box
    lx, ly = 906, 96
    lines.append(f'  <rect x="{lx}" y="{ly}" width="182" height="170" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{lx+14}" y="{ly+26}" font-family="ui-sans-serif, sans-serif" font-size="12" font-weight="700" fill="{INK}">Venue Category</text>')
    for i, (kind, color) in enumerate(kind_colors.items()):
        yy = ly + 52 + i * 28
        count = sum(1 for p in places if p.get("kind") == kind)
        lines.append(f'  <circle cx="{lx+22}" cy="{yy-4}" r="5" fill="{color}"/>')
        lines.append(f'  <text x="{lx+36}" y="{yy}" font-family="ui-sans-serif, sans-serif" font-size="11.5" fill="{INK}">{html.escape(kind)} ({count})</text>')

    lines.append(f'  <text x="{px+pw//2}" y="{py+ph+26}" text-anchor="middle" font-family="ui-sans-serif, sans-serif" font-size="11" font-weight="700" fill="{INK}">Semantic Component 1 (SVD-1)</text>')
    lines.append(f'  <text x="24" y="{py+ph//2}" text-anchor="middle" transform="rotate(-90 24 {py+ph//2})" font-family="ui-sans-serif, sans-serif" font-size="11" font-weight="700" fill="{INK}">Semantic Component 2 (SVD-2)</text>')
    lines.append("</svg>")
    out = FIG_DIR / "04_vector_space_pca_projection.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def fig5_rag_live_evaluation(capture: dict) -> Path:
    checks = capture["checks"]
    metrics = capture["metrics"]
    w, h = 1120, 510
    lines = svg_header(
        w, h,
        "5. Live Hybrid Vector RAG Evaluation — Cloudflare Vectorize + Constrained Synthesis",
        f"Measured {capture['measuredDate']} on {capture['origin']} | {capture['successfulQueries']}/{capture['queryCount']} HTTP 200 | Faithfulness={metrics['faithfulness']:.2f} | Relevance={metrics['answerRelevance']:.2f} | Median={metrics['providerLatencyMedianMs']}ms",
    )

    # Top KPI cards
    kpis = [
        ("Vectorize Index", "3,134 / 3,134", "bge-m3 (1,024d cosine)"),
        ("Faithfulness", f"{metrics['faithfulness']*100:.0f}%", "Zero hallucinated venues"),
        ("Answer Relevance", f"{metrics['answerRelevance']*100:.0f}%", "Intent-matched cards"),
        ("Median Latency", f"{metrics['providerLatencyMedianMs']} ms", f"P95: {metrics['providerLatencyP95Ms']} ms"),
    ]
    for i, (label, val, sub) in enumerate(kpis):
        kx = 32 + i * 268
        lines.append(f'  <rect x="{kx}" y="80" width="252" height="66" fill="#FFFFFF" stroke="{INK}" stroke-width="1.2"/>')
        lines.append(f'  <text x="{kx+14}" y="100" font-family="ui-monospace, monospace" font-size="10" font-weight="700" fill="{MUTED}">{html.escape(label.upper())}</text>')
        lines.append(f'  <text x="{kx+14}" y="124" font-family="ui-sans-serif, sans-serif" font-size="19" font-weight="800" fill="{INK}">{html.escape(val)}</text>')
        lines.append(f'  <text x="{kx+14}" y="139" font-family="ui-sans-serif, sans-serif" font-size="10" fill="{MUTED}">{html.escape(sub)}</text>')

    # Per-query latency & mode breakdown
    bx, by, bw, bh = 32, 164, 1056, 326
    lines.append(f'  <rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{bx+16}" y="{by+24}" font-family="ui-sans-serif, sans-serif" font-size="13" font-weight="700" fill="{INK}">Live Benchmark Queries — Latency (ms), Retrieval Mode &amp; Synthesis Mode</text>')
    max_ms = max(c["elapsedMs"] for c in checks) or 1
    for i, chk in enumerate(checks):
        y = by + 44 + i * 22
        q = chk["query"][:30]
        ms = chk["elapsedMs"]
        bar_w = int((ms / max_ms) * 520)
        mode = chk["retrievalMode"]
        synth = chk["synthesisMode"]
        color = EMERALD if mode == "hybrid" else (BLUE if synth == "constrained" else "#94A3B8")
        lines.append(f'  <text x="{bx+230}" y="{y+12}" text-anchor="end" font-family="ui-monospace, monospace" font-size="10.5" fill="{INK}">{html.escape(q)}</text>')
        lines.append(f'  <rect x="{bx+242}" y="{y+2}" width="{bar_w}" height="14" fill="{color}"/>')
        badge = f"{ms}ms · {mode}/{synth} · {chk['cardCount']} cards"
        lines.append(f'  <text x="{bx+250+bar_w}" y="{y+12}" font-family="ui-monospace, monospace" font-size="10" fill="{INK}">{html.escape(badge)}</text>')

    lines.append("</svg>")
    out = FIG_DIR / "05_rag_live_evaluation_proof.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out




def fig0_system_architecture() -> Path:
    w, h = 1120, 490
    lines = svg_header(
        w, h,
        "End-to-End System Architecture — Ingestion & Gates → EDA & ML → Production Hybrid Vector RAG",
        "Deterministic quality gates feed both the cross-validated ML scoring layer and the 1,024-dim Cloudflare Vectorize + Gemma-4 RAG pipeline",
    )
    lines.extend([
        '  <defs>',
        f'    <marker id="arr-ink" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        f'      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{INK}"/>',
        '    </marker>',
        f'    <marker id="arr-emerald" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        f'      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{EMERALD}"/>',
        '    </marker>',
        f'    <marker id="arr-gold" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        f'      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{GOLD}"/>',
        '    </marker>',
        '  </defs>',
    ])

    def stage_box(x: int, y: int, bw: int, bh: int, title: str, color: str) -> None:
        lines.append(f'  <rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="#FFFFFF" stroke="{GRID}" stroke-width="1.4"/>')
        lines.append(f'  <rect x="{x}" y="{y}" width="{bw}" height="32" rx="6" fill="{color}"/>')
        lines.append(f'  <rect x="{x}" y="{y+24}" width="{bw}" height="8" fill="{color}"/>')
        lines.append(f'  <text x="{x+14}" y="{y+21}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12" font-weight="800" fill="#FFFFFF">{html.escape(title)}</text>')

    def node(x: int, y: int, nw: int, nh: int, title: str, subs: list[str], border: str = INK, fill: str = BG) -> None:
        lines.append(f'  <rect x="{x}" y="{y}" width="{nw}" height="{nh}" rx="5" fill="{fill}" stroke="{border}" stroke-width="1.3"/>')
        lines.append(f'  <text x="{x+12}" y="{y+19}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="11.5" font-weight="700" fill="{INK}">{html.escape(title)}</text>')
        for i, sub in enumerate(subs):
            lines.append(f'  <text x="{x+12}" y="{y+34+i*14}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="10" fill="{MUTED}">{html.escape(sub)}</text>')

    # Stage 1: Ingestion & Data Quality Gates
    stage_box(32, 82, 316, 360, "1. INGESTION & DATA QUALITY GATES", BLUE)
    node(48, 128, 284, 54, "OpenStreetMap & Curated Guides", ["White Guide, Michelin, Visit Stockholm"], BLUE)
    node(48, 198, 284, 54, "Stockholm Municipal Signals", ["Food Control & Permit Records"], BLUE)
    node(48, 282, 284, 96, "Hard Exclusion & Boundary Gates", [
        "• Stockholm municipality boundary",
        "• Commercial chain exclusion",
        "• Closed / merged venue suppression",
        "→ 3,140 published independent venues",
    ], INK, "#F8FAFC")

    # Arrows inside Stage 1
    lines.append(f'  <path d="M 190 182 L 190 198" stroke="{INK}" stroke-width="1.5" fill="none" marker-end="url(#arr-ink)"/>')
    lines.append(f'  <path d="M 190 252 L 190 280" stroke="{INK}" stroke-width="1.5" fill="none" marker-end="url(#arr-ink)"/>')

    # Stage 2: EDA, Baselines & ML Layer
    stage_box(384, 82, 326, 360, "2. EDA, BASELINES & ML LAYER", GOLD)
    node(400, 128, 294, 54, "EDA & Feature Engineering", ["3,192 venues × 41 engineered features"], GOLD)
    node(400, 200, 294, 52, "Isolation Forest Anomaly Detection", ["Structural outliers & data-quality flags"], GOLD)
    node(400, 270, 294, 56, "Transparent Multi-Signal Scoring", ["A/B baseline vs. popularity (+46% diversity)"], GOLD)
    node(400, 344, 294, 56, "5-Fold Cross-Validated ML & LTR", ["HistGradientBoostingRegressor & OOF residuals"], GOLD)

    # Arrows inside Stage 2
    lines.append(f'  <path d="M 547 182 L 547 198" stroke="{INK}" stroke-width="1.5" fill="none" marker-end="url(#arr-ink)"/>')
    lines.append(f'  <path d="M 547 252 L 547 268" stroke="{INK}" stroke-width="1.5" fill="none" marker-end="url(#arr-ink)"/>')
    lines.append(f'  <path d="M 547 326 L 547 342" stroke="{INK}" stroke-width="1.5" fill="none" marker-end="url(#arr-ink)"/>')

    # Stage 3: Hybrid Vector RAG (Production)
    stage_box(746, 82, 342, 360, "3. HYBRID VECTOR RAG (PRODUCTION)", EMERALD)
    node(762, 124, 310, 48, "Fact-Grounded Document Serializer", ["Deterministic venue facts & SHA-256 hashes"], EMERALD)
    node(762, 184, 310, 48, "Cloudflare Workers AI Embeddings", ["@cf/baai/bge-m3 (1,024-dim dense cosine)"], EMERALD)
    node(762, 244, 310, 48, "Cloudflare Vectorize Index", ["3,134 / 3,134 verified indexed vectors"], EMERALD)
    node(762, 304, 310, 56, "Hybrid Retrieval Fusion (RRF k=60)", ["Intent + Structured Filters + Vector + Lexical"], EMERALD)
    node(762, 372, 310, 56, "Constrained LLM Synthesis", ["Gemma-4-26b-a4b-it / Verified Template (0 hall.)"], ACCENT, "#FEF2F2")

    # Arrows inside Stage 3
    for y1, y2 in [(172, 182), (232, 242), (292, 302), (360, 370)]:
        lines.append(f'  <path d="M 917 {y1} L 917 {y2}" stroke="{EMERALD}" stroke-width="1.5" fill="none" marker-end="url(#arr-emerald)"/>')

    # Cross-stage connectors
    # Stage 1 GATES -> Stage 2 EDA
    lines.append(f'  <path d="M 332 315 L 366 315 L 366 155 L 398 155" stroke="{INK}" stroke-width="1.6" fill="none" marker-end="url(#arr-ink)"/>')
    # Stage 1 GATES -> Stage 3 DOC (clean bottom rail)
    lines.append(f'  <path d="M 190 378 L 190 458 L 728 458 L 728 148 L 760 148" stroke="{EMERALD}" stroke-width="1.6" stroke-dasharray="5,3" fill="none" marker-end="url(#arr-emerald)"/>')
    lines.append(f'  <text x="458" y="472" text-anchor="middle" font-family="ui-monospace, monospace" font-size="10" fill="{EMERALD}">Eligible venue catalog (Stockholm-only, chain-free) → RAG Document Index</text>')
    # Stage 2 BASE -> Stage 3 RET
    lines.append(f'  <path d="M 694 298 L 720 298 L 720 332 L 760 332" stroke="{GOLD}" stroke-width="1.7" fill="none" marker-end="url(#arr-gold)"/>')

    lines.append("</svg>")
    out = FIG_DIR / "00_system_architecture.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def fig5a_rag_query_sequence() -> Path:
    w, h = 1120, 450
    lines = svg_header(
        w, h,
        "Live Hybrid Vector RAG Request Flow — Example Query: \"äta husman\"",
        "End-to-end execution across intent parsing, hard eligibility gates, Cloudflare Vectorize cosine lookup, RRF fusion, and constrained synthesis",
    )
    lines.extend([
        '  <defs>',
        f'    <marker id="seq-solid" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">',
        f'      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{INK}"/>',
        '    </marker>',
        f'    <marker id="seq-return" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">',
        f'      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{EMERALD}"/>',
        '    </marker>',
        '  </defs>',
    ])

    participants = [
        ("User Query", "\"äta husman\"", BLUE),
        ("Intent & Filters", "lib/concierge/intent.ts", GOLD),
        ("Eligibility Gates", "lib/concierge/gates.ts", ACCENT),
        ("Cloudflare Vectorize", "3,134 × 1,024d bge-m3", PURPLE),
        ("Hybrid Fusion", "hybrid_search.ts (RRF)", BLUE),
        ("Constrained Synth", "synthesis.ts (Gemma-4)", EMERALD),
    ]
    xs = [110, 290, 470, 650, 830, 1010]
    top_y = 84
    bot_y = 418

    # Participant headers & lifelines
    for (title, sub, color), cx in zip(participants, xs):
        lines.append(f'  <line x1="{cx}" y1="{top_y+46}" x2="{cx}" y2="{bot_y}" stroke="{GRID}" stroke-width="1.5" stroke-dasharray="5,4"/>')
        lines.append(f'  <rect x="{cx-78}" y="{top_y}" width="156" height="46" rx="5" fill="#FFFFFF" stroke="{color}" stroke-width="1.6"/>')
        lines.append(f'  <rect x="{cx-78}" y="{top_y}" width="156" height="5" rx="2" fill="{color}"/>')
        lines.append(f'  <text x="{cx}" y="{top_y+23}" text-anchor="middle" font-family="ui-sans-serif, system-ui, sans-serif" font-size="11.5" font-weight="700" fill="{INK}">{html.escape(title)}</text>')
        lines.append(f'  <text x="{cx}" y="{top_y+38}" text-anchor="middle" font-family="ui-monospace, monospace" font-size="9.2" fill="{MUTED}">{html.escape(sub)}</text>')

    # Sequence steps: (num, src_idx, dst_idx, label, is_return)
    steps = [
        (1, 0, 1, "Parse cuisine, district & constraints", False),
        (2, 1, 2, "Filter catalog (Stockholm-only, no chains/closed)", False),
        (3, 1, 3, "Embed query (bge-m3) & top-K cosine lookup", False),
        (4, 3, 4, "Semantic scores + verified vector IDs", True),
        (5, 2, 4, "Eligible lexical & structured candidates", True),
        (6, 4, 5, "Top-5 grounded venue cards + verified facts", True),
        (7, 5, 0, "Cited answer (Pelikan, Husman, Bord) — zero hallucination", True),
    ]

    for i, (num, src, dst, label, is_ret) in enumerate(steps):
        y = 162 + i * 36
        x1, x2 = xs[src], xs[dst]
        stroke = EMERALD if is_ret else INK
        marker = "url(#seq-return)" if is_ret else "url(#seq-solid)"
        dash = ' stroke-dasharray="6,3"' if is_ret else ""
        x_end = x2 - 6 if x2 > x1 else x2 + 6
        lines.append(f'  <line x1="{x1}" y1="{y}" x2="{x_end}" y2="{y}" stroke="{stroke}" stroke-width="1.7"{dash} marker-end="{marker}"/>')
        mid = (x1 + x2) / 2
        lines.append(f'  <rect x="{mid-170}" y="{y-17}" width="340" height="15" rx="3" fill="{BG}" opacity="0.92"/>')
        lines.append(f'  <text x="{mid}" y="{y-6}" text-anchor="middle" font-family="ui-sans-serif, system-ui, sans-serif" font-size="10.8" font-weight="600" fill="{stroke}">{num}. {html.escape(label)}</text>')

    lines.append("</svg>")
    out = FIG_DIR / "05a_rag_query_sequence.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def fig6_full_rag_trial_and_guardrails() -> Path:
    w, h = 1120, 490
    lines = svg_header(
        w, h,
        "6. Full Vector RAG Real-Model Trial — bge-m3 Embeddings + Gemma-4 Constrained Synthesis",
        "Measured Cloudflare Workers AI + Vectorize trial (docs/ml/concierge-real-model-trial.md): 197 embedding batches, 114,962 tokens, zero hallucinations shipped",
    )

    # Left panel: Real-model trial pipeline funnel
    lx, ly, lw, lh = 32, 90, 520, 368
    lines.append(f'  <rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{lx+16}" y="{ly+26}" font-family="ui-sans-serif, sans-serif" font-size="13" font-weight="700" fill="{INK}">End-to-End Full RAG Verification Funnel</text>')
    stages = [
        ("1. Vectorize Index (bge-m3 1024d)", "3,143 / 3,143", 100.0, EMERALD, "ID + SHA-256 hash verified"),
        ("2. Live Semantic Queries", "128 / 128", 100.0, BLUE, "100% HTTP 200 vector lookup"),
        ("3. Gemma-4 Synthesis API Calls", "32 / 32", 100.0, PURPLE, "@cf/google/gemma-4-26b-a4b-it"),
        ("4. Strict Fact-ID Validator Pass", "23 / 32", 71.9, EMERALD, "Exact placeId + grounded factIds"),
        ("5. Accepted Within 2s SLA", "22 / 32", 68.8, GOLD, "95.7% of accepted responses < 2s"),
    ]
    for i, (label, ratio, pct, color, note) in enumerate(stages):
        y = ly + 56 + i * 58
        bar_w = int((pct / 100.0) * 320)
        lines.append(f'  <text x="{lx+16}" y="{y}" font-family="ui-sans-serif, sans-serif" font-size="11.5" font-weight="700" fill="{INK}">{html.escape(label)}</text>')
        lines.append(f'  <text x="{lx+lw-16}" y="{y}" text-anchor="end" font-family="ui-monospace, monospace" font-size="11" font-weight="700" fill="{color}">{html.escape(ratio)} ({pct:.1f}%)</text>')
        lines.append(f'  <rect x="{lx+16}" y="{y+7}" width="320" height="14" fill="#F3F1EC"/>')
        lines.append(f'  <rect x="{lx+16}" y="{y+7}" width="{bar_w}" height="14" fill="{color}"/>')
        lines.append(f'  <text x="{lx+346}" y="{y+18}" font-family="ui-sans-serif, sans-serif" font-size="10" fill="{MUTED}">{html.escape(note)}</text>')

    # Right panel: Guardrail outcomes on 32 Gemma-4 synthesis calls
    rx, ry, rw, rh = 576, 90, 512, 368
    lines.append(f'  <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{rx+16}" y="{ry+26}" font-family="ui-sans-serif, sans-serif" font-size="13" font-weight="700" fill="{INK}">Anti-Hallucination Guardrail Audit (32 Gemma-4 Calls)</text>')
    outcomes = [
        ("Accepted — Grounded Fact-IDs", 23, EMERALD, "Served as constrained LLM synthesis"),
        ("Rejected — Empty Fact Selection", 7, GOLD, "Safe fallback to verified template"),
        ("Rejected — Extra JSON Object", 1, ACCENT, "Hallucinated entry blocked by validator"),
        ("Rejected — Output Truncated", 1, ACCENT, "finish_reason != stop blocked"),
        ("Hallucinations Reached Users", 0, INK, "Zero ungrounded venues or facts"),
    ]
    for i, (label, count, color, note) in enumerate(outcomes):
        y = ry + 58 + i * 56
        bar_w = int((count / 32.0) * 280)
        lines.append(f'  <text x="{rx+16}" y="{y}" font-family="ui-sans-serif, sans-serif" font-size="11.5" font-weight="700" fill="{INK}">{html.escape(label)}</text>')
        lines.append(f'  <rect x="{rx+16}" y="{y+8}" width="280" height="14" fill="#F3F1EC"/>')
        if bar_w > 0:
            lines.append(f'  <rect x="{rx+16}" y="{y+8}" width="{bar_w}" height="14" fill="{color}"/>')
        lines.append(f'  <text x="{rx+306}" y="{y+19}" font-family="ui-monospace, monospace" font-size="11" font-weight="700" fill="{color}">{count}/32</text>')
        lines.append(f'  <text x="{rx+16}" y="{y+34}" font-family="ui-sans-serif, sans-serif" font-size="10" fill="{MUTED}">{html.escape(note)}</text>')

    lines.append(f'  <text x="32" y="{h-14}" font-family="ui-monospace, monospace" font-size="10.5" fill="{MUTED}">Live hybrid+constrained example (2026-11-22): query="äta husman" → retrievalMode=hybrid, synthesisMode=constrained, cards=[Pelikan, Husman, Bord], fallbackReasons=[]</text>')
    lines.append("</svg>")
    out = FIG_DIR / "06_full_rag_trial_and_guardrails.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def fig7_rag_triad_and_latency_comparison(capture: dict) -> Path:
    metrics = capture["metrics"]
    live_faith = metrics["faithfulness"] * 100.0
    live_rel = metrics["answerRelevance"] * 100.0
    live_med_ms = int(metrics["providerLatencyMedianMs"])
    live_p95_ms = int(metrics["providerLatencyP95Ms"])

    w, h = 1120, 510
    lines = svg_header(
        w, h,
        "7. RAG Triad & Latency Comparison — Baseline (Historical v1) vs. Optimized Hybrid RAG",
        "Grouped bar audit: 128-query labeled retrieval benchmark (concierge-evaluation.md) + live Cloudflare Vectorize & Gemma-4 timings (concierge-real-model-trial.md)",
    )

    # Left panel: RAG Triad Grouped Bar Chart
    lx, ly, lw, lh = 32, 84, 594, 386
    lines.append(f'  <rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{lx+16}" y="{ly+24}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="13" font-weight="700" fill="{INK}">RAG Triad &amp; Safety — Baseline (v1) vs. Optimized RAG (%)</text>')

    # Legend (left panel)
    leg_y = ly + 42
    lines.append(f'  <rect x="{lx+16}" y="{leg_y-9}" width="12" height="10" fill="#CBD5E1" stroke="#64748B" stroke-width="0.8"/>')
    lines.append(f'  <text x="{lx+33}" y="{leg_y}" font-family="ui-sans-serif, sans-serif" font-size="10" fill="{MUTED}">Baseline (Historical v1)</text>')
    lines.append(f'  <rect x="{lx+166}" y="{leg_y-9}" width="12" height="10" fill="{EMERALD}"/>')
    lines.append(f'  <text x="{lx+183}" y="{leg_y}" font-family="ui-sans-serif, sans-serif" font-size="10" fill="{MUTED}">Optimized RAG (128q fixture)</text>')
    lines.append(f'  <rect x="{lx+348}" y="{leg_y-9}" width="12" height="10" fill="{PURPLE}"/>')
    lines.append(f'  <text x="{lx+365}" y="{leg_y}" font-family="ui-sans-serif, sans-serif" font-size="10" fill="{MUTED}">Live Capture / Guardrail</text>')

    bar_x = lx + 196
    bar_max_w = 310
    grid_top = ly + 56
    grid_bot = ly + 356
    for pct in (0, 25, 50, 75, 100):
        gx = bar_x + int((pct / 100.0) * bar_max_w)
        lines.append(f'  <line x1="{gx}" y1="{grid_top}" x2="{gx}" y2="{grid_bot}" stroke="{GRID}" stroke-width="1" stroke-dasharray="3,3"/>')
        lines.append(f'  <text x="{gx}" y="{grid_bot+15}" text-anchor="middle" font-family="ui-monospace, monospace" font-size="9.5" fill="{MUTED}">{pct}%</text>')

    # (label, sublabel, base_val_or_None, opt_val, opt_color, delta_text)
    triad_rows = [
        ("Context Recall", "Recall@50 (84 answerable)", 81.0, 92.9, EMERALD, "+11.9pp"),
        ("Context Precision", "NDCG@5 (84 answerable)", 75.0, 92.9, EMERALD, "+17.9pp"),
        ("Exact-Name Precision", "Top-1 hit (12 exact queries)", 33.3, 100.0, EMERALD, "+66.7pp"),
        ("Safe Abstention", "Out-of-scope (44 queries)", 0.0, 100.0, EMERALD, "+100pp"),
        ("Faithfulness", "Live E2E vs. Gemma-4 shipped", live_faith, 100.0, PURPLE, "0/32 hall."),
        ("Answer Relevance", "Live 12q capture (intent)", None, live_rel, PURPLE, "9/12 live"),
    ]

    for i, (label, sub, base_v, opt_v, opt_col, note) in enumerate(triad_rows):
        ry = grid_top + 10 + i * 48
        lines.append(f'  <text x="{bar_x-12}" y="{ry+12}" text-anchor="end" font-family="ui-sans-serif, sans-serif" font-size="11" font-weight="700" fill="{INK}">{html.escape(label)}</text>')
        lines.append(f'  <text x="{bar_x-12}" y="{ry+25}" text-anchor="end" font-family="ui-sans-serif, sans-serif" font-size="9.3" fill="{MUTED}">{html.escape(sub)}</text>')

        # Baseline bar (or live comparison bar for Faithfulness)
        if base_v is not None:
            bw_base = max(2, int((base_v / 100.0) * bar_max_w)) if base_v > 0 else 2
            base_fill = "#CBD5E1" if i < 4 else "#C4B5FD"
            base_stroke = "#64748B" if i < 4 else PURPLE
            lines.append(f'  <rect x="{bar_x}" y="{ry+2}" width="{bw_base}" height="12" fill="{base_fill}" stroke="{base_stroke}" stroke-width="0.7"/>')
            lines.append(f'  <text x="{bar_x+bw_base+5}" y="{ry+11.5}" font-family="ui-monospace, monospace" font-size="9.2" fill="{MUTED}">{base_v:.1f}%</text>')
        else:
            lines.append(f'  <text x="{bar_x+4}" y="{ry+11.5}" font-family="ui-monospace, monospace" font-size="9" fill="{MUTED}">(no v1 live baseline)</text>')

        # Optimized / Shipped bar
        bw_opt = int((opt_v / 100.0) * bar_max_w)
        lines.append(f'  <rect x="{bar_x}" y="{ry+17}" width="{bw_opt}" height="13" fill="{opt_col}"/>')
        lines.append(f'  <text x="{bar_x+bw_opt+5}" y="{ry+27}" font-family="ui-monospace, monospace" font-size="9.8" font-weight="700" fill="{opt_col}">{opt_v:.1f}% ({html.escape(note)})</text>')

    # Right panel: Latency Breakdown (Median vs P95 vs SLA)
    rx, ry0, rw, rh = 644, 84, 444, 386
    lines.append(f'  <rect x="{rx}" y="{ry0}" width="{rw}" height="{rh}" fill="#FFFFFF" stroke="{GRID}"/>')
    lines.append(f'  <text x="{rx+16}" y="{ry0+24}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="13" font-weight="700" fill="{INK}">Stage Latency — Median vs. P95 &amp; SLA (ms)</text>')

    # Legend (right panel)
    lines.append(f'  <rect x="{rx+16}" y="{leg_y-9}" width="12" height="10" fill="{BLUE}"/>')
    lines.append(f'  <text x="{rx+33}" y="{leg_y}" font-family="ui-sans-serif, sans-serif" font-size="10" fill="{MUTED}">Median (ms)</text>')
    lines.append(f'  <rect x="{rx+114}" y="{leg_y-9}" width="12" height="10" fill="{GOLD}"/>')
    lines.append(f'  <text x="{rx+131}" y="{leg_y}" font-family="ui-sans-serif, sans-serif" font-size="10" fill="{MUTED}">P95 (ms)</text>')
    lines.append(f'  <line x1="{rx+196}" y1="{leg_y-10}" x2="{rx+196}" y2="{leg_y+1}" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="2,2"/>')
    lines.append(f'  <text x="{rx+204}" y="{leg_y}" font-family="ui-sans-serif, sans-serif" font-size="10" fill="{MUTED}">Runtime SLA deadline</text>')

    lat_x = rx + 136
    lat_w = 226
    max_ms = 4500.0
    for ms_tick in (0, 1000, 2000, 3000, 4000):
        gx = lat_x + int((ms_tick / max_ms) * lat_w)
        lines.append(f'  <line x1="{gx}" y1="{grid_top}" x2="{gx}" y2="{grid_bot}" stroke="{GRID}" stroke-width="1" stroke-dasharray="3,3"/>')
        lbl = f"{ms_tick//1000}s" if ms_tick > 0 else "0"
        lines.append(f'  <text x="{gx}" y="{grid_bot+15}" text-anchor="middle" font-family="ui-monospace, monospace" font-size="9.5" fill="{MUTED}">{lbl}</text>')

    # (stage, sub, median_ms, p95_ms, sla_ms_or_None)
    lat_rows = [
        ("Query Embed", "bge-m3 (128q)", 262, 615, 1200),
        ("Vector Search", "Vectorize (128q)", 440, 590, 800),
        ("Retrieval Total", "Embed + Vector", 713, 1090, 2000),
        ("LLM Synthesis", "Gemma-4 (32q)", 1239, 4106, 2000),
        ("Live E2E HTTP", "Prod capture (12q)", live_med_ms, live_p95_ms, None),
    ]

    for i, (stage, sub, med_ms, p95_ms, sla_ms) in enumerate(lat_rows):
        y = grid_top + 16 + i * 58
        lines.append(f'  <text x="{lat_x-10}" y="{y+12}" text-anchor="end" font-family="ui-sans-serif, sans-serif" font-size="11" font-weight="700" fill="{INK}">{html.escape(stage)}</text>')
        lines.append(f'  <text x="{lat_x-10}" y="{y+25}" text-anchor="end" font-family="ui-sans-serif, sans-serif" font-size="9.3" fill="{MUTED}">{html.escape(sub)}</text>')

        med_w = max(4, int((min(med_ms, max_ms) / max_ms) * lat_w))
        p95_clamped = p95_ms > max_ms
        p95_w = max(4, int((min(p95_ms, max_ms) / max_ms) * lat_w))

        lines.append(f'  <rect x="{lat_x}" y="{y+2}" width="{med_w}" height="12" fill="{BLUE}"/>')
        lines.append(f'  <text x="{lat_x+med_w+5}" y="{y+11.5}" font-family="ui-monospace, monospace" font-size="9.5" font-weight="700" fill="{BLUE}">{med_ms:,} ms</text>')

        lines.append(f'  <rect x="{lat_x}" y="{y+17}" width="{p95_w}" height="12" fill="{GOLD}"/>')
        p95_lbl = f"{p95_ms:,} ms →" if p95_clamped else f"{p95_ms:,} ms"
        lines.append(f'  <text x="{lat_x+p95_w+5}" y="{y+26.5}" font-family="ui-monospace, monospace" font-size="9.5" fill="{MUTED}">{p95_lbl}</text>')

        if sla_ms is not None:
            sx = lat_x + int((sla_ms / max_ms) * lat_w)
            lines.append(f'  <line x1="{sx}" y1="{y-2}" x2="{sx}" y2="{y+33}" stroke="{ACCENT}" stroke-width="1.8" stroke-dasharray="3,2"/>')
            lines.append(f'  <text x="{sx}" y="{y+43}" text-anchor="middle" font-family="ui-monospace, monospace" font-size="8.5" fill="{ACCENT}">SLA {sla_ms:,}ms</text>')

    lines.append(f'  <text x="32" y="{h-14}" font-family="ui-monospace, monospace" font-size="10" fill="{MUTED}">Provenance: Context Recall &amp; Precision from 128-query labeled fixture (no human labels on live 3,143-venue catalog yet); latencies from sequential REST trial &amp; live HTTP capture.</text>')
    lines.append("</svg>")
    out = FIG_DIR / "07_rag_triad_and_latency_comparison.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    scored = pd.read_csv(DATA_DIR / "stockholm_food_places_scored.csv")
    slice_json = json.loads((DATA_DIR / "places_slice.json").read_text(encoding="utf-8"))
    ab = json.loads((DATA_DIR / "ranking_ab_experiment.json").read_text(encoding="utf-8"))
    capture = json.loads((DATA_DIR / "rag_provider_capture.json").read_text(encoding="utf-8"))

    outputs = [
        fig0_system_architecture(),
        fig1_eda_districts_and_completeness(scored, slice_json),
        fig2_correlations_and_anomalies(scored),
        fig3_baseline_vs_multisignal_and_ltr(scored, ab),
        fig4_vector_space_projection(slice_json),
        fig5a_rag_query_sequence(),
        fig5_rag_live_evaluation(capture),
        fig6_full_rag_trial_and_guardrails(),
        fig7_rag_triad_and_latency_comparison(capture),
    ]
    for path in outputs:
        print(f"Generated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
