"""
app/views/charts.py

Plotly chart builders shared across dashboard pages (UI/UX redesign v2).
Presentation only: constructor-colored marks, hover templates, tidy axes.
No business logic, no API calls, no src/ imports — callers pass prepared
frames. All charts render with width="stretch" (the non-deprecated form of
use_container_width=True).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.views.components import OTHER_COLOR, WINNER_COLOR, constructor_color


def win_share_bar(frame: pd.DataFrame, winner_id: int | None = None,
                  color_by_constructor: bool = True, height: int = 420) -> None:
    """Horizontal win-share bars, one per driver, best rank on top.

    Expects columns: label, win_probability, driver_id, predicted_rank,
    and optionally constructor_name (drives the bar colors). The actual
    winner keeps a thick F1-red outline so it stays visually distinct even
    among same-team bars.
    """
    plot = frame.iloc[::-1]                       # rank 1 renders topmost
    if color_by_constructor and "constructor_name" in plot.columns:
        colors = [constructor_color(t) for t in plot["constructor_name"]]
        teams = plot["constructor_name"].fillna("").tolist()
    else:
        colors = OTHER_COLOR
        teams = [""] * len(plot)
    is_winner = (plot["driver_id"] == winner_id if winner_id is not None
                 else pd.Series(False, index=plot.index))

    fig = go.Figure(go.Bar(
        x=plot["win_probability"],
        y=plot["label"],
        orientation="h",
        marker_color=colors,
        marker_line_color=[WINNER_COLOR if w else "rgba(0,0,0,0)" for w in is_winner],
        marker_line_width=[4 if w else 0 for w in is_winner],
        text=[f"{v:.1%}" for v in plot["win_probability"]],
        textposition="outside",
        customdata=list(zip(teams, plot["predicted_rank"], strict=True)),
        hovertemplate=("<b>%{y}</b><br>%{customdata[0]}<br>"
                       "Win share %{x:.1%} · Model rank #%{customdata[1]}"
                       "<extra></extra>"),
    ))
    fig.update_layout(
        xaxis_title="Win share (normalized within race)",
        xaxis_tickformat=".0%", height=height, showlegend=False,
        margin=dict(l=10, r=48, t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")


def trend_line(frame: pd.DataFrame, x: str, y: str, *, title: str | None = None,
               x_label: str | None = None, y_label: str | None = None,
               invert_y: bool = False, color: str | None = None,
               height: int = 300) -> None:
    """Single-series line+marker trend. invert_y=True puts P1 at the top —
    use it for any position/rank axis."""
    fig = go.Figure(go.Scatter(
        x=frame[x], y=frame[y], mode="lines+markers",
        line_color=color or OTHER_COLOR,
        hovertemplate=(f"{x_label or x} %{{x}}<br>"
                       f"{y_label or y}: %{{y}}<extra></extra>"),
    ))
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label,
        height=height, showlegend=False,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
    )
    if invert_y:
        fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")


def standings_bar(frame: pd.DataFrame, name_col: str, value_col: str,
                  color_col: str | None = None, height: int = 360) -> None:
    """Horizontal standings bars, leader on top; color_col (a constructor
    display-name column) drives brand colors when given."""
    plot = frame.iloc[::-1]                       # leader renders topmost
    colors = ([constructor_color(c) for c in plot[color_col]]
              if color_col and color_col in plot.columns else OTHER_COLOR)
    fig = go.Figure(go.Bar(
        x=plot[value_col], y=plot[name_col].astype(str), orientation="h",
        marker_color=colors,
        text=plot[value_col], textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x}<extra></extra>",
    ))
    fig.update_layout(
        height=height, showlegend=False, xaxis_title=value_col.capitalize(),
        margin=dict(l=10, r=48, t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")


def histogram(series: pd.Series, x_label: str, nbins: int = 12,
              height: int = 300, percent_axis: bool = False) -> None:
    """Distribution histogram for a numeric series."""
    fig = go.Figure(go.Histogram(
        x=series, nbinsx=nbins, marker_color=OTHER_COLOR,
        hovertemplate="%{x}: %{y} races<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=x_label, yaxis_title="Races", height=height,
        showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
    )
    if percent_axis:
        fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
