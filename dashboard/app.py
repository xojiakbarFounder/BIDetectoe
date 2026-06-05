from __future__ import annotations

import datetime as dt
import html
import os
import sys
import time
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from utils import (
    api_base_url,
    fetch_hourly_stats,
    fetch_latest_events,
    fetch_live_state,
    fetch_peak_hour,
    fetch_today_counts,
    send_chat,
)


st.set_page_config(
    page_title="Live People Counter",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
:root {
  --ink: #111827;
  --muted: #64748b;
  --line: #dbe4ee;
  --paper: #f7fafc;
  --panel: #ffffff;
  --green: #10b981;
  --red: #f43f5e;
  --amber: #f59e0b;
  --cyan: #0891b2;
}

.stApp {
  background:
    linear-gradient(180deg, #f8fbff 0%, #eef5f3 46%, #f8fafc 100%);
  color: var(--ink);
}

section[data-testid="stSidebar"] {
  background: #0f172a;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  color: #e5eef8 !important;
}

.block-container {
  padding-top: 1.25rem;
  padding-bottom: 1.5rem;
  max-width: 1380px;
}

.topbar {
  border-bottom: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: end;
  padding: 6px 0 18px;
}

.title-block h1 {
  color: #0f172a;
  font-size: 34px;
  letter-spacing: 0;
  line-height: 1.05;
  margin: 0;
}

.title-block p {
  color: var(--muted);
  margin: 8px 0 0;
}

.status-chip {
  align-items: center;
  background: #e8fff6;
  border: 1px solid #b7f7df;
  border-radius: 999px;
  color: #065f46;
  display: inline-flex;
  font-weight: 700;
  gap: 8px;
  padding: 8px 12px;
  white-space: nowrap;
}

.status-chip.offline {
  background: #fff1f2;
  border-color: #fecdd3;
  color: #9f1239;
}

.dot {
  background: currentColor;
  border-radius: 50%;
  display: inline-block;
  height: 9px;
  width: 9px;
}

.section-label {
  color: #334155;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: .08em;
  margin: 4px 0 10px;
  text-transform: uppercase;
}

.metric-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin: 16px 0 8px;
}

.metric-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 5px solid var(--cyan);
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, .06);
  min-height: 116px;
  padding: 16px;
}

.metric-card.green { border-left-color: var(--green); }
.metric-card.red { border-left-color: var(--red); }
.metric-card.amber { border-left-color: var(--amber); }
.metric-card.blue { border-left-color: #2563eb; }
.metric-card.pink { border-left-color: #db2777; }

.metric-label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
}

.metric-value {
  color: #0f172a;
  font-size: 34px;
  font-weight: 800;
  letter-spacing: 0;
  line-height: 1;
  margin-top: 12px;
}

.metric-note {
  color: var(--muted);
  font-size: 13px;
  margin-top: 10px;
}

.stream-shell {
  background: #07111f;
  border: 1px solid #17243a;
  border-radius: 8px;
  box-shadow: 0 14px 34px rgba(15, 23, 42, .18);
  min-height: 410px;
  overflow: hidden;
  position: relative;
}

.stream-shell img {
  aspect-ratio: 16 / 9;
  display: block;
  height: auto;
  object-fit: cover;
  width: 100%;
}

.stream-shell iframe {
  aspect-ratio: 16 / 9;
  border: 0;
  display: block;
  height: auto;
  width: 100%;
}

.stream-shell .stream-help {
  background: rgba(15, 23, 42, .82);
  border-top: 1px solid rgba(226, 232, 240, .12);
  bottom: 0;
  color: #cbd5e1;
  font-size: 12px;
  left: 0;
  padding: 8px 14px;
  position: absolute;
  right: 0;
}

.stream-empty {
  align-items: center;
  color: #cbd5e1;
  display: flex;
  min-height: 410px;
  justify-content: center;
  text-align: center;
}

.stream-badge {
  background: rgba(15, 23, 42, .82);
  border: 1px solid rgba(226, 232, 240, .18);
  border-radius: 999px;
  color: #f8fafc;
  font-size: 12px;
  font-weight: 800;
  left: 14px;
  letter-spacing: .06em;
  padding: 7px 10px;
  position: absolute;
  text-transform: uppercase;
  top: 14px;
}

.event-strip {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, .06);
  height: 100%;
  padding: 14px;
}

.event-row {
  align-items: center;
  border-bottom: 1px solid #edf2f7;
  display: grid;
  gap: 10px;
  grid-template-columns: 52px 1fr auto;
  padding: 10px 0;
}

.event-row:last-child { border-bottom: 0; }

.event-id {
  background: #eef2ff;
  border-radius: 6px;
  color: #3730a3;
  font-weight: 800;
  padding: 6px 8px;
  text-align: center;
}

.event-dir {
  color: #0f172a;
  font-weight: 800;
}

.event-time {
  color: var(--muted);
  font-size: 12px;
}

.chat-user, .chat-bot {
  border-radius: 8px;
  clear: both;
  margin: 8px 0;
  max-width: 78%;
  padding: 10px 12px;
}

.chat-user {
  background: #0f766e;
  color: white;
  float: right;
}

.chat-bot {
  background: #e2e8f0;
  color: #0f172a;
  float: left;
}

div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
}

button[kind="primary"] {
  background: #0f766e !important;
  border: 1px solid #0f766e !important;
}

@media (max-width: 900px) {
  .topbar {
    align-items: start;
    flex-direction: column;
  }
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .stream-shell, .stream-empty {
    min-height: 260px;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def _fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _fmt_time(value: object) -> str:
    if not value:
        return "-"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "-"
    try:
        parsed = parsed.tz_localize(None)
    except TypeError:
        parsed = parsed.tz_convert(None)
    return parsed.strftime("%H:%M:%S")


def _period_label(selected_date: dt.date) -> str:
    if selected_date == dt.date.today():
        return "Bugun"
    return selected_date.strftime("%d.%m.%Y")


def _metric_card(label: str, value: object, note: str, tone: str = "") -> None:
    st.markdown(
        f"""
<div class="metric-card {tone}">
  <div class="metric-label">{html.escape(label)}</div>
  <div class="metric-value">{_fmt_int(value)}</div>
  <div class="metric-note">{html.escape(note)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _youtube_embed_url(url: str) -> str | None:
    parsed = urlparse(url)
    video_id = ""
    if parsed.netloc.endswith("youtu.be"):
        video_id = parsed.path.strip("/")
    elif "youtube.com" in parsed.netloc:
        if parsed.path.startswith("/watch"):
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith("/live/") or parsed.path.startswith("/embed/"):
            video_id = parsed.path.rstrip("/").split("/")[-1]

    if not video_id:
        return None
    return (
        f"https://www.youtube.com/embed/{video_id}"
        "?autoplay=1&mute=1&controls=1&playsinline=1&rel=0"
    )


def _render_stream(source: str) -> None:
    if source == "YouTube live":
        embed_url = _youtube_embed_url(settings.youtube_stream_url)
        if embed_url:
            st.markdown(
                f"""
<div class="stream-shell">
  <div class="stream-badge">YouTube live</div>
  <iframe src="{html.escape(embed_url)}" allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe>
  <div class="stream-help">{html.escape(settings.youtube_stream_url)}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            return

    feed_url = f"{api_base_url()}/video/feed?client=streamlit"
    st.markdown(
        f"""
<div class="stream-shell">
  <div class="stream-badge">YOLO live</div>
  <img src="{html.escape(feed_url)}" alt="Live detection stream">
  <div class="stream-help">MJPEG live feed: pipeline frame yuborishi bilan video uzluksiz yangilanadi.</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _events_html(events: list[dict]) -> str:
    if not events:
        return '<div class="metric-note">Hali crossing event yoq.</div>'

    rows = []
    for event in events[:7]:
        direction = event.get("direction", "-")
        dir_label = "Kirdi" if direction == "in" else "Chiqdi"
        dir_color = "#047857" if direction == "in" else "#be123c"
        object_class = html.escape(str(event.get("object_class", "person")).title())
        timestamp = _fmt_time(event.get("timestamp"))
        tracker_id = html.escape(str(event.get("tracker_id", "-")))
        event_id = html.escape(str(event.get("id", "-")))
        rows.append(
            f"""
<div class="event-row">
  <div class="event-id">#{event_id}</div>
  <div>
    <div class="event-dir" style="color:{dir_color}">{dir_label}</div>
    <div class="event-time">{object_class} | Tracker {tracker_id}</div>
  </div>
  <div class="event-time">{timestamp}</div>
</div>
"""
        )
    return "\n".join(rows)


with st.sidebar:
    st.header("Boshqaruv")
    selected_date = st.date_input(
        "Kun",
        value=dt.date.today(),
        max_value=dt.date.today(),
    )
    auto_refresh = st.checkbox("Statistika auto refresh", value=False)
    refresh_seconds = st.slider("Refresh soniya", 2, 15, 3)
    stream_source = st.radio(
        "Video manbasi",
        ["YOLO annotated", "YouTube live"],
        horizontal=False,
    )

    st.divider()
    st.caption("API")
    st.code(api_base_url(), language="text")
    st.caption("Excel export")
    st.code(settings.excel_export_dir, language="text")

    if st.button("Yangilash", use_container_width=True):
        st.rerun()


date_str = selected_date.isoformat()
period = _period_label(selected_date)
live = fetch_live_state()
counts = fetch_today_counts(date_str)
hourly = fetch_hourly_stats(date_str)
events = fetch_latest_events(limit=30)
peak = fetch_peak_hour(days=7)

live_total = live.get("total_count", 0)
category_counts = counts.get("category_counts") or live.get("category_counts", {})
person_total = category_counts.get("person", 0)
car_total = category_counts.get("car", 0)
motorcycle_total = category_counts.get("motorcycle", 0)
live_status = live.get("timestamp") not in (None, "-", "вЂ”")
status_class = "" if live_status else "offline"
status_text = "Online" if live_status else "Offline"

st.markdown(
    f"""
<div class="topbar">
  <div class="title-block">
    <h1>Live People Counter</h1>
    <p>People, cars and motorcycles live detector with Excel-ready event history</p>
  </div>
  <div class="status-chip {status_class}">
    <span class="dot"></span>{status_text}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    _metric_card("People", person_total, f"{period} crossing", "green")
with m2:
    _metric_card("Cars", car_total, f"{period} crossing", "blue")
with m3:
    _metric_card("Motorcycles", motorcycle_total, f"{period} crossing", "pink")
with m4:
    _metric_card("Total", counts.get("total", live_total), f"FPS {live.get('fps', 0.0)}", "amber")
st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns([1.65, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Kamera oqimi</div>', unsafe_allow_html=True)
    _render_stream(stream_source)

with right:
    st.markdown('<div class="section-label">Hozirgi holat</div>', unsafe_allow_html=True)
    l1, l2 = st.columns(2)
    l1.metric("Active tracks", _fmt_int(live.get("active_tracks", 0)))
    l2.metric("FPS", live.get("fps", 0.0))
    st.markdown(
        f"""
<div class="event-strip">
  <div class="section-label">Songgi otishlar</div>
  {_events_html(events)}
</div>
""",
        unsafe_allow_html=True,
    )

st.divider()

chart_col, table_col = st.columns([1.35, 1], gap="large")

with chart_col:
    st.markdown(f'<div class="section-label">{period} soatlik grafik</div>', unsafe_allow_html=True)
    if hourly:
        df = pd.DataFrame(hourly)
        df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
        df = df.dropna(subset=["hour"]).sort_values("hour")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df["hour"],
                y=df["in_count"],
                name="Kirdi",
                marker_color="#10b981",
                hovertemplate="%{x|%H:%M}<br>Kirdi: %{y}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=df["hour"],
                y=df["out_count"],
                name="Chiqdi",
                marker_color="#f43f5e",
                hovertemplate="%{x|%H:%M}<br>Chiqdi: %{y}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["hour"],
                y=df["total_count"],
                name="Jami",
                mode="lines+markers",
                line=dict(color="#0891b2", width=3),
                marker=dict(size=7),
                hovertemplate="%{x|%H:%M}<br>Jami: %{y}<extra></extra>",
            )
        )
        fig.update_layout(
            barmode="group",
            height=390,
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(l=10, r=10, t=22, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,.78)",
            xaxis=dict(title="", gridcolor="#e5edf5"),
            yaxis=dict(title="Odamlar", gridcolor="#e5edf5"),
            font=dict(color="#0f172a"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"{period} uchun soatlik malumot topilmadi.")

with table_col:
    st.markdown('<div class="section-label">Kunlik tafsilotlar</div>', unsafe_allow_html=True)
    if hourly:
        display = df[["hour", "in_count", "out_count", "total_count"]].copy()
        display["hour"] = display["hour"].dt.strftime("%H:%M")
        display.columns = ["Soat", "Kirdi", "Chiqdi", "Jami"]
        st.dataframe(display, use_container_width=True, hide_index=True, height=260)
    else:
        st.dataframe(
            pd.DataFrame(columns=["Soat", "Kirdi", "Chiqdi", "Jami"]),
            use_container_width=True,
            hide_index=True,
            height=260,
        )

    if peak:
        peak_hour = pd.to_datetime(peak.get("hour"), errors="coerce")
        peak_label = "-" if pd.isna(peak_hour) else peak_hour.strftime("%d.%m %H:00")
        st.success(
            f"7 kunlik peak: {peak_label}, jami {peak.get('total_count', 0)} ta odam."
        )
    else:
        st.info("7 kunlik peak hali hisoblanmagan.")

st.divider()

events_col, chat_col = st.columns([1.05, 1], gap="large")

with events_col:
    st.markdown('<div class="section-label">Event jurnali</div>', unsafe_allow_html=True)
    if events:
        event_df = pd.DataFrame(events)
        event_df["Vaqt"] = pd.to_datetime(event_df["timestamp"], errors="coerce")
        event_df["Vaqt"] = event_df["Vaqt"].dt.strftime("%d.%m.%Y %H:%M:%S")
        event_df["Yonalish"] = event_df["direction"].map({"in": "Kirdi", "out": "Chiqdi"}).fillna("-")
        event_df["Kategoriya"] = event_df["object_class"].fillna("person").str.title()
        event_df = event_df.rename(
            columns={"id": "ID", "tracker_id": "Tracker"}
        )
        st.dataframe(
            event_df[["ID", "Kategoriya", "Tracker", "Yonalish", "Vaqt"]],
            use_container_width=True,
            hide_index=True,
            height=360,
        )
    else:
        st.info("Eventlar hali kelmadi.")

with chat_col:
    st.markdown('<div class="section-label">AI yordamchi</div>', unsafe_allow_html=True)

    prompts = [
        "Bugun nechta odam otdi?",
        "Peak soat qaysi?",
        "Kirdi va chiqdini solishtir",
        "Bugungi holatni qisqa xulosa qil",
    ]
    p1, p2 = st.columns(2)
    for idx, prompt in enumerate(prompts):
        target = p1 if idx % 2 == 0 else p2
        with target:
            if st.button(prompt, key=f"prompt_{idx}", use_container_width=True):
                st.session_state["pending_chat"] = prompt

    with st.container():
        for turn in st.session_state.chat_history[-8:]:
            css = "chat-user" if turn["role"] == "user" else "chat-bot"
            content = html.escape(turn["content"])
            st.markdown(f'<div class="{css}">{content}</div>', unsafe_allow_html=True)
        st.markdown('<div style="clear:both"></div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        default_msg = st.session_state.pop("pending_chat", "")
        msg = st.text_input(
            "Savol",
            value=default_msg,
            placeholder="Savolingizni yozing",
            label_visibility="collapsed",
        )
        sent = st.form_submit_button("Yuborish", type="primary", use_container_width=True)

    clear = st.button("Suhbatni tozalash", use_container_width=True)

    if sent and msg.strip():
        st.session_state.chat_history.append({"role": "user", "content": msg.strip()})
        with st.spinner("Javob tayyorlanmoqda..."):
            reply = send_chat(msg.strip(), st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if clear:
        st.session_state.chat_history = []
        st.rerun()

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
