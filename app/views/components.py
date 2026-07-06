"""
app/views/components.py

Reusable, presentation-only UI building blocks shared across dashboard pages
(UI/UX redesign — see context/decisions.md for the rationale). Every helper
here is a thin wrapper over native Streamlit widgets (st.container(border=),
st.metric(border=), st.badge) — no custom CSS/HTML injection, so styling
stays theme-aware (light/dark) for free and there is no injection surface.

Consumed by app/views/{home,race_center,driver_explorer,season_analytics,
insights}.py. This module renders only — it never calls the API itself
(that stays in app/views/common.py) and never imports src/.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

# F1-flavored accent colors — purely cosmetic grouping, not a data source.
WINNER_COLOR = "#E10600"     # F1 red — actual winner / hero accents
OTHER_COLOR = "#4C78A8"

# Constructor brand colors (hex, for Plotly charts) keyed by lowercase
# substring of the Ergast constructor name — substring-tolerant so renames
# like "RB F1 Team" or "Alfa Romeo Racing" still resolve. Order matters:
# more specific keys ("red bull") before shorter overlapping ones.
CONSTRUCTOR_COLORS = {
    "red bull": "#3671C6",
    "ferrari": "#E8002D",
    "mercedes": "#27F4D2",
    "mclaren": "#FF8000",
    "alpine": "#FF87BC",
    "aston martin": "#229971",
    "williams": "#64C4FF",
    "haas": "#B6BABD",
    "alphatauri": "#6692FF",
    "rb f1": "#6692FF",
    "toro rosso": "#2B4562",
    "alfa romeo": "#C92D4B",
    "sauber": "#52E252",
    "renault": "#FFF500",
    "racing point": "#F596C8",
    "force india": "#F596C8",
    "lotus": "#B8A15A",
    "brawn": "#B8FD6E",
    "caterham": "#048646",
    "marussia": "#6E0000",
    "manor": "#323232",
    "jaguar": "#0A5C2F",
    "toyota": "#CC0000",
    "bmw": "#0066B2",
}
DEFAULT_CONSTRUCTOR_COLOR = OTHER_COLOR

# Emoji dot per team for card captions — st.badge only supports preset
# colors, so cards get a nearest-color dot while charts use the real hex.
_TEAM_DOTS = {
    "red bull": "🔵", "ferrari": "🔴", "mercedes": "🟢", "mclaren": "🟠",
    "alpine": "🟣", "aston martin": "🟢", "williams": "🔵", "haas": "⚪",
    "alphatauri": "🔵", "rb f1": "🔵", "toro rosso": "🔵", "alfa romeo": "🔴",
    "sauber": "🟢", "renault": "🟡", "racing point": "🟣", "force india": "🟣",
    "lotus": "🟡", "brawn": "🟡",
}


def constructor_color(name: str | None) -> str:
    """Brand hex for a constructor display name; neutral fallback."""
    if not name:
        return DEFAULT_CONSTRUCTOR_COLOR
    low = str(name).lower()
    for key, color in CONSTRUCTOR_COLORS.items():
        if key in low:
            return color
    return DEFAULT_CONSTRUCTOR_COLOR


def constructor_dot(name: str | None) -> str:
    """Team-colored emoji dot for card captions."""
    if not name:
        return "🏎"
    low = str(name).lower()
    for key, dot in _TEAM_DOTS.items():
        if key in low:
            return dot
    return "🏎"


def confidence_label(probability: float) -> str:
    """Fan-friendly wording for a win share — storytelling stand-in for raw
    calibrated probabilities on user-facing pages."""
    if probability >= 0.50:
        return "Strong favorite"
    if probability >= 0.30:
        return "Clear favorite"
    if probability >= 0.20:
        return "Slight edge"
    return "Wide open"


def page_header(title: str, icon: str, subtitle: str | None = None) -> None:
    """Consistent page title + optional one-line subtitle, used at the top
    of every page instead of ad hoc st.title()/st.markdown() calls."""
    st.title(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)


def stat_tile(label: str, value: str, help: str | None = None,   # noqa: A002
              delta: str | None = None, delta_color: str = "normal") -> None:
    """One bordered metric card. Thin wrapper so every page gets identical
    tile styling (border, spacing) without repeating st.metric(border=True)."""
    st.metric(label, value, delta=delta, delta_color=delta_color,
              help=help, border=True)


def stat_row(items: Sequence[dict]) -> None:
    """A row of equal-width stat_tile()s. Each item: {label, value, help?,
    delta?, delta_color?}. Replaces the repeated `col1, col2, col3 =
    st.columns(3); col1.metric(...)` pattern that existed on every page."""
    if not items:
        return
    for col, item in zip(st.columns(len(items)), items, strict=True):
        with col:
            stat_tile(
                item["label"], item["value"],
                help=item.get("help"), delta=item.get("delta"),
                delta_color=item.get("delta_color", "normal"),
            )


def status_badge(ok: bool, ok_text: str = "Healthy",
                  bad_text: str = "Degraded") -> None:
    """Green/red pill badge for a boolean health state."""
    if ok:
        st.badge(ok_text, icon=":material/check_circle:", color="green")
    else:
        st.badge(bad_text, icon=":material/error:", color="red")


def hit_miss_badge(hit: bool | None) -> None:
    """Pill badge for whether the model's top pick actually won."""
    if hit is None:
        st.badge("Outcome pending", color="gray")
    elif hit:
        st.badge("Model picked the winner", icon=":material/check_circle:", color="green")
    else:
        st.badge("Model missed the winner", icon=":material/close:", color="orange")


def confidence_bar(probability: float, label: str = "Confidence") -> None:
    """0-1 probability rendered as a progress bar + percentage caption —
    the storytelling-friendly stand-in for exposing raw/calibrated model
    internals on user-facing pages."""
    st.progress(min(max(probability, 0.0), 1.0), text=f"{label}: {probability:.0%}")


def driver_label(prediction: dict) -> str:
    """'Name (Team)' from a /predictions DriverPrediction dict, falling back
    to the numeric id when the display-name lookup is unavailable."""
    name = prediction.get("driver_name") or f"driver {prediction['driver_id']}"
    team = prediction.get("constructor_name")
    return f"{name} ({team})" if team else name


def favorite_card(prediction: dict, year: int, round_: int,
                  subtitle: str | None = None) -> None:
    """Hero card for the model's #1 pick — the first thing a fan should
    see on the Race Center page."""
    team = prediction.get("constructor_name")
    with st.container(border=True):
        st.caption(subtitle or f"{year} · Round {round_} · Model favorite")
        st.subheader(f"🏆 {driver_label(prediction)}")
        if team:
            st.caption(f"{constructor_dot(team)} {team}")
        confidence_bar(prediction["win_probability"], label="Win share")
        st.caption(f"Confidence: **{confidence_label(prediction['win_probability'])}**")


def driver_card(prediction: dict, grid: int | None = None,
                quali: int | None = None, trend: int | None = None,
                is_winner: bool = False) -> None:
    """Compact contender card: rank + name, team dot, grid/quali line, win
    share, rank-trend arrow vs the previous round, actual-winner badge."""
    name = prediction.get("driver_name") or f"driver {prediction['driver_id']}"
    team = prediction.get("constructor_name")
    with st.container(border=True):
        st.markdown(f"**#{prediction['predicted_rank']} {name}**")
        if team:
            st.caption(f"{constructor_dot(team)} {team}")
        line = []
        if grid is not None:
            line.append("Grid PL" if grid == 0 else f"Grid P{grid}")
        if quali is not None:
            line.append(f"Quali P{quali}")
        if line:
            st.caption(" · ".join(line))
        st.markdown(f"**{prediction['win_probability']:.1%}** win share")
        if trend is not None and trend != 0:
            arrow, color = ("▲", "green") if trend > 0 else ("▼", "orange")
            st.badge(f"{arrow} {abs(trend)} vs last race", color=color)
        if is_winner:
            st.badge("Actual winner", icon=":material/emoji_events:", color="green")


def reason_badges(reasons: Sequence[str]) -> None:
    """'Why this prediction' factors as green check badges."""
    if not reasons:
        return
    cols = st.columns(min(4, len(reasons)))
    for col, reason in zip(cols, reasons[:4], strict=False):
        with col:
            st.badge(reason, icon=":material/check:", color="green")
    for reason in reasons[4:]:
        st.badge(reason, icon=":material/check:", color="green")


def race_header(label: str, season: int, round_: int,
                circuit: str | None = None, date: str | None = None) -> None:
    """Grand Prix header block: name + season/round/circuit/date chips."""
    st.header(label)
    chips = [f"🗓 {season} · Round {round_}"]
    if circuit:
        chips.append(f"📍 {circuit}")
    if date:
        chips.append(f"📅 {date}")
    st.caption("  ·  ".join(chips))


def podium_row(predictions: Sequence[dict], winner_id: int | None) -> None:
    """Top-3 predicted finishers as three side-by-side cards (medal icons),
    each flagging the actual winner if known."""
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(min(3, len(predictions)))
    for medal, pred, col in zip(medals, predictions[:3], cols, strict=False):
        with col, st.container(border=True):
            st.markdown(f"**{medal} {driver_label(pred)}**")
            st.caption(f"{pred['win_probability']:.1%} win share")
            if winner_id is not None and pred["driver_id"] == winner_id:
                st.badge("Actual winner", icon=":material/emoji_events:", color="green")


def badge_card(label: str, text: str, color: str = "blue",
               icon: str | None = None) -> None:
    """Bordered card matching stat_tile's footprint, for a categorical status
    value (e.g. serving stage, health) shown as a colored badge rather than
    a plain metric number."""
    with st.container(border=True):
        st.caption(label)
        st.badge(text, color=color, icon=icon)


def empty_state(message: str, icon: str = "ℹ️") -> None:
    """Friendly placeholder for 'nothing to show' states (no data for this
    filter combination) instead of a blank page or a stack trace."""
    st.info(f"{icon} {message}")
