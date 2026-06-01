from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime as dt
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.database import get_day_counts, get_hourly_stats, get_peak_hour, get_latest_events

st.set_page_config(
    page_title="Kuzatuv Tizimi",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.chat-msg-user { background:#2563eb; color:#fff; border-radius:12px 12px 2px 12px;
                 padding:10px 14px; margin:6px 0; max-width:80%; float:right; clear:both; }
.chat-msg-bot  { background:#374151; color:#f9fafb; border-radius:12px 12px 12px 2px;
                 padding:10px 14px; margin:6px 0; max-width:80%; float:left; clear:both; }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

# ── Sarlavha ──────────────────────────────────────────────────────────────────

col_title, col_btn = st.columns([6, 1])
with col_title:
    st.title("Kuzatuv Analitika Tizimi")
    st.caption("YOLOv8 + ByteTrack · Walworth Road London")
with col_btn:
    st.write("")
    if st.button("Yangilash"):
        st.rerun()

st.divider()

# ── Kun filtri ────────────────────────────────────────────────────────────────

tanlangan_kun = st.date_input(
    "Kun tanlang",
    value=dt.date.today(),
    max_value=dt.date.today(),
)
bugun    = tanlangan_kun == dt.date.today()
kun_nomi = "Bugun" if bugun else tanlangan_kun.strftime("%d-%b-%Y")

# ═══════════════════════════════════════════════════════════════════════════════
# 1-QISM — Statistika
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader(f"{kun_nomi} Statistikasi")

counts = get_day_counts(tanlangan_kun)

in_val    = counts.get("in_count", 0)
out_val   = counts.get("out_count", 0)
total_val = counts.get("total", 0)

k1, k2, k3 = st.columns(3)
k1.metric("🟢 Kirdi",  in_val)
k2.metric("🔴 Chiqdi", out_val)
k3.metric("⚪ Jami",   total_val)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 2-QISM — Soatlik grafik
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader(f" {kun_nomi} — Soatlik Ko'rsatkich")

hourly = get_hourly_stats(tanlangan_kun)

if hourly:
    df = pd.DataFrame(hourly)
    df["hour"] = pd.to_datetime(df["hour"])
    df = df.sort_values("hour")

    # Qaysi ustunlar DB da mavjudligini aniqlaymiz
    has_in  = df["in_count"].sum() > 0
    has_out = df["out_count"].sum() > 0

    tab1, tab2, tab3 = st.tabs([" Ustunli", " Chiziqli", " Jadval"])

    with tab1:
        fig = px.bar(
            df, x="hour", y=["in_count", "out_count"],
            barmode="group",
            labels={"value": "Odamlar", "hour": "Soat", "variable": ""},
            color_discrete_map={"in_count": "#22c55e", "out_count": "#ef4444"},
            title=f"Soatlik harakat — {kun_nomi}",
        )
        fig.for_each_trace(lambda t: t.update(
            name="Kirdi" if t.name == "in_count" else "Chiqdi"
        ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f9fafb",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["hour"], y=df["total_count"], mode="lines+markers",
            name="Jami", line=dict(color="#6366f1", width=2)))
        fig2.add_trace(go.Scatter(
            x=df["hour"], y=df["in_count"], mode="lines",
            name="Kirdi", line=dict(color="#22c55e", width=1, dash="dot")))
        fig2.add_trace(go.Scatter(
            x=df["hour"], y=df["out_count"], mode="lines",
            name="Chiqdi", line=dict(color="#ef4444", width=1, dash="dot")))
        fig2.update_layout(
            xaxis_title="Soat", yaxis_title="Odamlar",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f9fafb",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        disp = df[["hour", "in_count", "out_count", "total_count"]].copy()
        disp.columns = ["Soat", "Kirdi", "Chiqdi", "Jami"]
        disp["Soat"] = df["hour"].dt.strftime("%H:%M")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    peak = get_peak_hour(days=7)
    if peak:
        ph = dt.datetime.strptime(peak["hour"], "%Y-%m-%d %H:%M:%S")
        st.info(
            f"**7 kunlik eng gavjum soat:** {ph.strftime('%d-%b %H:00')} — "
            f"{peak['total_count']} ta odam"
        )
else:
    st.info(f"{kun_nomi} uchun ma'lumot yo'q.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 3-QISM — So'nggi o'tishlar
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("So'nggi O'tishlar")

events = get_latest_events(limit=20)
if events:
    edf = pd.DataFrame(events)
    edf["vaqt"] = pd.to_datetime(edf["timestamp"]).dt.tz_localize(None)
    edf["yo'nalish"] = edf["direction"].map({"in": "🟢 Kirdi", "out": "🔴 Chiqdi"})
    st.dataframe(
        edf[["id", "tracker_id", "yo'nalish", "vaqt"]].rename(columns={
            "id": "ID", "tracker_id": "Kishi ID",
            "yo'nalish": "Yo'nalish", "vaqt": "Vaqt",
        }),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("Hali o'tish qayd etilmagan.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 4-QISM — AI Chatbot
# ═══════════════════════════════════════════════════════════════════════════════

from utils import send_chat

st.subheader("AI Yordamchisi")
st.caption("Kuzatuv ma'lumotlari haqida so'rang.")

s_cols = st.columns(3)
savollar = [
    "Bugun nechta odam o'tdi?",
    "Eng gavjum soat qaysi edi?",
    "Bugungi faollikni xulosa qil",
    "Kirdi va chiqdini solishtir",
    "Oxirgi soatdagi harakat",
    "Haftalik statistika",
]
for i, s in enumerate(savollar):
    with s_cols[i % 3]:
        if st.button(s, key=f"s_{i}"):
            st.session_state["pending_chat"] = s

with st.container():
    for turn in st.session_state.chat_history:
        css = "chat-msg-user" if turn["role"] == "user" else "chat-msg-bot"
        st.markdown(f'<div class="{css}">{turn["content"]}</div>', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    msg  = st.text_input("Savol", value=st.session_state.pop("pending_chat", ""),
                         placeholder="Masalan: Bugun nechta odam o'tdi?",
                         label_visibility="collapsed")
    send = st.form_submit_button("Yuborish ➤")

if send and msg.strip():
    st.session_state.chat_history.append({"role": "user", "content": msg})
    with st.spinner("Javob tayyorlanmoqda…"):
        reply = send_chat(msg, st.session_state.chat_history[:-1])
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()

if st.button(" Suhbatni tozalash"):
    st.session_state.chat_history = []
    st.rerun()

time.sleep(0.1)
