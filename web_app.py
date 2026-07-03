"""
FateAutomata Kiosk — Remote Web Deployment Layer
=================================================
All data is pulled from Supabase — no local CSV or Pi files needed.
The Knowledge Base/ FAISS folder must be present alongside this file.

Required Supabase tables:
  attendance  (student_name, timestamp, synced, lanyard_compliant, dresscode_compliant)
  metrics     (timestamp, face_extract_ms, lanyard_ms, dress_ms, end_to_end_ms,
               cpu_percent, ram_percent, temp_c, raw_dress_detected, smoothed_dress_detected)

Run:  streamlit run web_app.py
"""

# ── 1. Compatibility Shim (MUST be before any Ragas/LangChain imports) ────────
import sys

try:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai
except ImportError:
    # Failsafe stub to prevent Ragas from crashing on Streamlit Cloud
    from types import ModuleType
    _stub = ModuleType("langchain_community.chat_models.vertexai")
    class _ChatVertexAI:  
        pass
    _stub.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

# ── 2. Standard Imports ────────────────────────────────────────────────────────
import os
import math
import logging
import nest_asyncio
from datetime import datetime, date

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
import daily_report as dr

from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig

nest_asyncio.apply()
load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

# ── 3. Constants & Page Config ─────────────────────────────────────────────────
REFRESH_TTL  = 15
PAGE_SIZE    = 25
METRICS_ROWS = 500

st.set_page_config(
    page_title="FateAutomata — Remote Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #f8fafc; border-radius: 10px; padding: 12px 16px;
}
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 9999px;
    font-size: 12px; font-weight: 600;
}
.badge-red    { background: #fee2e2; color: #991b1b; }
.badge-orange { background: #ffedd5; color: #9a3412; }
.badge-green  { background: #dcfce7; color: #166534; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE CLIENT & DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
    if not url or not key:
        st.error("❌ Missing SUPABASE_URL or SUPABASE_KEY in Streamlit secrets.")
        st.stop()
    # Normalize URL (remove trailing /v1 or /rest/v1)
    if "/rest/v1" in url:
        url = url.split("/rest/v1")[0]
    elif "/v1" in url:
        url = url.split("/v1")[0]
    url = url.rstrip("/")
    return create_client(url, key)

def _normalize_bools(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["lanyard_compliant", "dresscode_compliant", "synced"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: str(v).strip().lower() in ("true", "1", "yes", "t")
                if not isinstance(v, bool) else v
            )
    return df

@st.cache_data(ttl=REFRESH_TTL)
def load_attendance() -> pd.DataFrame:
    sb = get_supabase()
    try:
        resp = (
            sb.table("attendance")
            .select("student_name, timestamp, synced, lanyard_compliant, dresscode_compliant")
            .order("timestamp", desc=True)
            .limit(2000)
            .execute()
        )
        if not resp.data:
            return _empty_attendance()
        df = pd.DataFrame(resp.data)
        df = _normalize_bools(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        df = df.rename(columns={
            "student_name":       "Student_Name",
            "timestamp":          "Timestamp",
            "synced":             "Cloud_Synced",
            "lanyard_compliant":  "Lanyard_Compliant",
            "dresscode_compliant":"DressCode_Compliant",
        })
        return df
    except Exception as e:
        st.warning(f"Could not load attendance: {e}")
        return _empty_attendance()

def _empty_attendance() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "Student_Name", "Timestamp", "Cloud_Synced",
        "Lanyard_Compliant", "DressCode_Compliant",
    ])

@st.cache_data(ttl=REFRESH_TTL)
def load_metrics() -> pd.DataFrame:
    sb = get_supabase()
    try:
        resp = (
            sb.table("metrics")
            .select("*")
            .order("timestamp", desc=True)
            .limit(METRICS_ROWS)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# AI COMPONENT LOADER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_ai_components():
    cohere_key = os.environ.get("COHERE_API_KEY") or st.secrets.get("COHERE_API_KEY", "")
    groq_key   = os.environ.get("GROQ_API_KEY")   or st.secrets.get("GROQ_API_KEY", "")

    embeddings   = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_key)
    vector_store = FAISS.load_local(
        "Knowledge Base", embeddings, allow_dangerous_deserialization=True
    )
    generator_llm = ChatCohere(model="command-r-08-2024", cohere_api_key=cohere_key)
    judge_llm     = ChatGroq(
        model_name="llama-3.3-70b-versatile", api_key=groq_key, temperature=0.0
    )
    return vector_store, generator_llm, judge_llm


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "mapua_logo.png",
        width=80,
    )
    st.title("FateAutomata")
    st.caption("Remote Kiosk Dashboard")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Live Dashboard", "📈 Analytics", "👤 Student Profiles",
         "🖥️ Hardware Monitor", "🤖 AI Assistant", "📋 Compliance Report"],
        label_visibility="collapsed",
    )
    st.divider()

    with st.expander("⚙️ Settings"):
        auto_refresh = st.toggle("Auto-refresh (15 s)", value=True)

    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last load: {datetime.now().strftime('%H:%M:%S')} UTC")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Live Dashboard":
    st.header("📊 Live Attendance & Compliance Dashboard")

    df = load_attendance()
    today_utc = pd.Timestamp(date.today(), tz="UTC")
    today_df  = df[df["Timestamp"] >= today_utc] if not df.empty else df

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Logs (All Time)", len(df))
    with k2:
        st.metric("Logs Today", len(today_df))
    with k3:
        synced = int(df["Cloud_Synced"].sum()) if not df.empty else 0
        st.metric("Synced to Supabase", synced)
    with k4:
        missing_id    = int((~df["Lanyard_Compliant"]).sum())    if not df.empty       else 0
        today_missing = int((~today_df["Lanyard_Compliant"]).sum()) if not today_df.empty else 0
        st.metric("🚨 Missing IDs (All)", missing_id,
                  delta=f"{today_missing} today", delta_color="inverse")
    with k5:
        dress_v    = int((~df["DressCode_Compliant"]).sum())       if not df.empty       else 0
        today_dress= int((~today_df["DressCode_Compliant"]).sum()) if not today_df.empty else 0
        st.metric("👕 Dress Violations (All)", dress_v,
                  delta=f"{today_dress} today", delta_color="inverse")

    st.divider()
    col_feed, col_chart = st.columns([1.4, 1])

    with col_feed:
        st.subheader("Recent Violation Events")
        violations = (
            df[~(df["Lanyard_Compliant"] & df["DressCode_Compliant"])].head(15)
            if not df.empty else df
        )
        if violations.empty:
            st.success("✅ No violations on record — all students compliant.")
        else:
            for _, row in violations.iterrows():
                ts_str = row["Timestamp"].strftime("%b %d, %H:%M:%S") if pd.notna(row["Timestamp"]) else "—"
                flags  = []
                if not row["Lanyard_Compliant"]:
                    flags.append('<span class="badge badge-red">No Lanyard</span>')
                if not row["DressCode_Compliant"]:
                    flags.append('<span class="badge badge-orange">Dress Code</span>')
                flag_html   = " ".join(flags)
                synced_icon = "☁️" if row["Cloud_Synced"] else "💾"
                st.markdown(
                    f"""<div style="background:#fff7ed;border-left:3px solid #f97316;
                        padding:8px 12px;border-radius:6px;margin-bottom:6px;color:#451a03;">
                        <b>{row['Student_Name']}</b> &nbsp; {flag_html}
                        <span style="float:right;color:#000000;font-size:12px">
                            {synced_icon} {ts_str}
                        </span></div>""",
                    unsafe_allow_html=True,
                )

    with col_chart:
        st.subheader("Today's Compliance Breakdown")
        if not today_df.empty:
            full_ok   = int((today_df["Lanyard_Compliant"] & today_df["DressCode_Compliant"]).sum())
            no_lan    = int((~today_df["Lanyard_Compliant"] & today_df["DressCode_Compliant"]).sum())
            dress_bad = int((today_df["Lanyard_Compliant"] & ~today_df["DressCode_Compliant"]).sum())
            both_bad  = int((~today_df["Lanyard_Compliant"] & ~today_df["DressCode_Compliant"]).sum())
            pie_df = pd.DataFrame({
                "Category": ["Fully Compliant","No Lanyard","Dress Violation","Both Violations"],
                "Count":    [full_ok, no_lan, dress_bad, both_bad],
            })
            fig_pie = px.pie(
                pie_df, values="Count", names="Category", hole=0.45,
                color="Category",
                color_discrete_map={
                    "Fully Compliant":"#22c55e","No Lanyard":"#ef4444",
                    "Dress Violation":"#f97316","Both Violations":"#7c3aed",
                },
            )
            fig_pie.update_layout(margin=dict(t=10,b=10), height=260, legend=dict(font_size=12))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No entries for today yet.")

        st.subheader("Arrivals per Hour (Today)")
        if not today_df.empty:
            h_counts = (
                today_df.copy()
                .assign(Hour=today_df["Timestamp"].dt.hour)
                .groupby("Hour").size().reset_index(name="Count")
            )
            fig_hr = px.bar(h_counts, x="Hour", y="Count",
                            color_discrete_sequence=["#3b82f6"], height=200)
            fig_hr.update_layout(margin=dict(t=5,b=5,l=5,r=5),
                                 xaxis_title="Hour (UTC)", yaxis_title="")
            st.plotly_chart(fig_hr, use_container_width=True)

    st.divider()
    st.subheader("Full Attendance Log")
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search_q = st.text_input("🔍 Filter by student name", placeholder="e.g. CORPUZ")
    with col_f2:
        if not df.empty and "Timestamp" in df.columns:
            available_dates = sorted(df["Timestamp"].dropna().dt.date.unique(), reverse=True)
            date_options = ["All Days"] + [d.strftime("%Y-%m-%d") for d in available_dates]
        else:
            date_options = ["All Days"]
        selected_date_str = st.selectbox("📅 Filter by day", date_options, index=0)

    filtered = df.copy()
    if search_q:
        filtered = filtered[filtered["Student_Name"].astype(str).str.contains(search_q, case=False)]
    if selected_date_str != "All Days":
        sel_date = datetime.strptime(selected_date_str, "%Y-%m-%d").date()
        filtered = filtered[filtered["Timestamp"].dt.date == sel_date]

    total_pages = max(1, math.ceil(len(filtered) / PAGE_SIZE))
    pg1, pg2, _ = st.columns([1, 2, 6])
    with pg1:
        page_num = st.number_input("Page", min_value=1, max_value=total_pages,
                                   value=1, step=1, label_visibility="collapsed")
    with pg2:
        st.caption(f"Page {page_num} of {total_pages}  ({len(filtered)} rows)")

    start_i = (page_num - 1) * PAGE_SIZE
    page_df = filtered.iloc[start_i : start_i + PAGE_SIZE].copy()
    for col in ["Cloud_Synced", "Lanyard_Compliant", "DressCode_Compliant"]:
        if col in page_df.columns:
            page_df[col] = page_df[col].apply(lambda v: "✅" if v else "❌")
    if "Timestamp" in page_df.columns:
        page_df["Timestamp"] = page_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(page_df, use_container_width=True, hide_index=True)

    if auto_refresh:
        import time
        time.sleep(REFRESH_TTL)
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Analytics":
    st.header("📈 Compliance Analytics")

    df = load_attendance()
    if df.empty:
        st.info("No data in Supabase yet.")
        st.stop()

    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()
    dc1, dc2 = st.columns(2)
    with dc1:
        start_d = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    with dc2:
        end_d = st.date_input("To",   value=max_date, min_value=min_date, max_value=max_date)

    mask = (df["Timestamp"].dt.date >= start_d) & (df["Timestamp"].dt.date <= end_d)
    fdf  = df[mask].copy()
    if fdf.empty:
        st.warning("No records in selected range.")
        st.stop()

    st.caption(f"Showing **{len(fdf)}** records — {start_d} → {end_d}")
    st.divider()

    st.subheader("Violation Trend (Daily)")
    trend = (
        fdf.assign(Date=fdf["Timestamp"].dt.date,
                   Lanyard_V=~fdf["Lanyard_Compliant"],
                   Dress_V=~fdf["DressCode_Compliant"])
        .groupby("Date")[["Lanyard_V","Dress_V"]].sum().reset_index()
        .rename(columns={"Lanyard_V":"Lanyard Violations","Dress_V":"Dress Violations"})
    )
    fig_trend = px.line(trend, x="Date", y=["Lanyard Violations","Dress Violations"],
                        markers=True, color_discrete_sequence=["#ef4444","#f97316"])
    fig_trend.update_layout(legend_title="", yaxis_title="Count", height=300)
    st.plotly_chart(fig_trend, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most Violations by Student")
        sv = (
            fdf.assign(Any_V=~(fdf["Lanyard_Compliant"] & fdf["DressCode_Compliant"]))
            .groupby("Student_Name")
            .agg(Total_Logs=("Timestamp","count"), Violations=("Any_V","sum"))
            .reset_index()
        )
        sv["Compliance Rate"] = (1 - sv["Violations"] / sv["Total_Logs"].clip(lower=1)).round(2)
        sv = sv.sort_values("Violations", ascending=False).head(10)
        fig_sv = px.bar(sv, x="Violations", y="Student_Name", orientation="h",
                        color="Violations", color_continuous_scale="Reds", height=320)
        fig_sv.update_layout(yaxis_title="", coloraxis_showscale=False,
                              margin=dict(l=5,r=5,t=10,b=5))
        st.plotly_chart(fig_sv, use_container_width=True)

    with c2:
        st.subheader("Arrival Heatmap (Hour × Day)")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        hm = fdf.assign(Hour=fdf["Timestamp"].dt.hour,
                        DayOfWeek=fdf["Timestamp"].dt.day_name())
        hm_pivot = hm.groupby(["DayOfWeek","Hour"]).size().reset_index(name="Count")
        hm_pivot["DayOfWeek"] = pd.Categorical(hm_pivot["DayOfWeek"], categories=day_order, ordered=True)
        hm_full = (hm_pivot.sort_values("DayOfWeek")
                   .pivot(index="DayOfWeek", columns="Hour", values="Count").fillna(0))
        fig_hm = px.imshow(hm_full, color_continuous_scale="Blues", aspect="auto", height=320,
                           labels=dict(x="Hour (UTC)", y="", color="Arrivals"))
        fig_hm.update_layout(margin=dict(l=5,r=5,t=10,b=5))
        st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("Per-Student Compliance Summary")
    sv["Compliance Rate"] = sv["Compliance Rate"].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(sv.rename(columns={"Student_Name":"Student"}),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — STUDENT PROFILES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👤 Student Profiles":
    st.header("👤 Student Profiles")

    df = load_attendance()
    if df.empty:
        st.info("No attendance data yet.")
        st.stop()

    students = sorted(df["Student_Name"].dropna().unique().tolist())
    selected = st.selectbox("Select a student", students)
    sdf = df[df["Student_Name"] == selected].copy()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Appearances", len(sdf))
    with k2:
        st.metric("Lanyard Violations", int((~sdf["Lanyard_Compliant"]).sum()))
    with k3:
        st.metric("Dress Violations", int((~sdf["DressCode_Compliant"]).sum()))
    with k4:
        rate = (sdf["Lanyard_Compliant"] & sdf["DressCode_Compliant"]).mean()
        st.metric("Compliance Rate", f"{rate*100:.1f}%")

    st.divider()
    st.subheader(f"Scan History — {selected}")

    def status_label(row):
        lan, drs = row["Lanyard_Compliant"], row["DressCode_Compliant"]
        if lan and drs:   return "✅ Compliant"
        if not lan and not drs: return "🔴 Both Violations"
        if not lan:       return "🟠 No Lanyard"
        return "🟡 Dress Code"

    sdf_disp = sdf.copy()
    sdf_disp["Status"]    = sdf_disp.apply(status_label, axis=1)
    sdf_disp["Timestamp"] = sdf_disp["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    sdf_disp["Synced"]    = sdf_disp["Cloud_Synced"].apply(lambda v: "☁️" if v else "💾")
    st.dataframe(sdf_disp[["Timestamp","Status","Synced"]],
                 use_container_width=True, hide_index=True)

    if len(sdf) >= 2:
        st.subheader("Compliance Trend")
        ts_trend = (
            sdf.assign(Date=sdf["Timestamp"].dt.date,
                       OK=(sdf["Lanyard_Compliant"] & sdf["DressCode_Compliant"]))
            .groupby("Date")["OK"].mean().reset_index()
            .rename(columns={"OK":"Compliance Rate"})
        )
        ts_trend["Compliance Rate"] *= 100
        fig_ts = px.line(ts_trend, x="Date", y="Compliance Rate",
                         markers=True, color_discrete_sequence=["#22c55e"], height=220)
        fig_ts.update_layout(yaxis_range=[0,105], yaxis_title="%",
                              margin=dict(t=5,b=5))
        st.plotly_chart(fig_ts, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HARDWARE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🖥️ Hardware Monitor":
    st.header("🖥️ Raspberry Pi 5 — Hardware Telemetry")
    st.caption("Data pushed from `metrics_log.csv` on the Pi to the `metrics` Supabase table.")

    mdf = load_metrics()
    if mdf.empty:
        st.info("No metrics data in Supabase yet. Make sure `recognize.py` is syncing its metrics.")
        st.stop()

    latest = mdf.iloc[-1]

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        cpu = latest.get("cpu_percent", None)
        st.metric("CPU %", f"{float(cpu):.1f}%" if cpu is not None else "—")
    with k2:
        ram = latest.get("ram_percent", None)
        st.metric("RAM %", f"{float(ram):.1f}%" if ram is not None else "—")
    with k3:
        temp = latest.get("temp_c", None)
        st.metric("CPU Temp", f"{float(temp):.1f}°C" if temp is not None else "—",
                  delta_color="inverse" if temp and float(temp) > 70 else "normal")
    with k4:
        e2e = latest.get("end_to_end_ms", None)
        st.metric("End-to-End", f"{float(e2e):.1f} ms" if e2e is not None else "—")
    with k5:
        fps_val = 1000 / float(e2e) if e2e and float(e2e) > 0 else 0
        st.metric("Effective FPS", f"{fps_val:.1f}")

    st.divider()

    st.subheader("Pipeline Latency Over Time (ms)")
    lat_cols  = ["face_extract_ms","lanyard_ms","dress_ms","end_to_end_ms"]
    available = [c for c in lat_cols if c in mdf.columns]
    if available:
        lat_df = mdf[["timestamp"] + available].set_index("timestamp")
        for c in lat_df.columns:
            lat_df[c] = pd.to_numeric(lat_df[c], errors="coerce")
        lat_df = lat_df.dropna(how="all")
        if not lat_df.empty:
            fig_lat = px.line(lat_df, color_discrete_sequence=["#3b82f6","#22c55e","#f97316","#7c3aed"])
            fig_lat.update_layout(legend_title="Stage", yaxis_title="ms",
                                  height=300, margin=dict(t=5,b=5))
            st.plotly_chart(fig_lat, use_container_width=True)

    hw1, hw2 = st.columns(2)
    with hw1:
        st.subheader("CPU & RAM Utilization")
        hw_df = mdf[["timestamp","cpu_percent","ram_percent"]].copy()
        for c in ["cpu_percent","ram_percent"]:
            hw_df[c] = pd.to_numeric(hw_df[c], errors="coerce")
        hw_df = hw_df.dropna().set_index("timestamp")
        if not hw_df.empty:
            fig_hw = px.area(hw_df, color_discrete_sequence=["#3b82f6","#f97316"], height=250)
            fig_hw.update_layout(legend_title="", yaxis_title="%",
                                 yaxis_range=[0,100], margin=dict(t=5,b=5))
            st.plotly_chart(fig_hw, use_container_width=True)

    with hw2:
        st.subheader("CPU Temperature (°C)")
        temp_df = mdf[["timestamp","temp_c"]].copy()
        temp_df["temp_c"] = pd.to_numeric(temp_df["temp_c"], errors="coerce")
        temp_df = temp_df.dropna().set_index("timestamp")
        if not temp_df.empty:
            fig_t = px.line(temp_df, color_discrete_sequence=["#ef4444"], height=250)
            fig_t.add_hline(y=75, line_dash="dash", line_color="#f97316",
                            annotation_text="Throttle threshold (75°C)")
            fig_t.update_layout(yaxis_title="°C", showlegend=False, margin=dict(t=5,b=5))
            st.plotly_chart(fig_t, use_container_width=True)

    if "raw_dress_detected" in mdf.columns and "smoothed_dress_detected" in mdf.columns:
        st.subheader("Dress Code Detection: Raw vs Smoothed")
        sm_df = mdf[["timestamp","raw_dress_detected","smoothed_dress_detected"]].tail(200).copy()
        for c in ["raw_dress_detected","smoothed_dress_detected"]:
            sm_df[c] = pd.to_numeric(sm_df[c], errors="coerce")
        sm_df = sm_df.dropna().set_index("timestamp")
        if not sm_df.empty:
            fig_sm = px.line(sm_df, color_discrete_sequence=["#f87171","#1d4ed8"], height=220)
            fig_sm.update_layout(legend_title="", yaxis_title="Detected (1=Yes)",
                                 yaxis_range=[-0.1,1.3], margin=dict(t=5,b=5))
            st.plotly_chart(fig_sm, use_container_width=True)
            st.caption("Temporal smoothing suppresses single-frame false positives "
                       "from the 1-class YOLO dataset bias.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 AI Assistant":
    st.header("🤖 Mapúa Prefect of Discipline — AI Assistant")
    st.markdown(
        "Ask any question about university policies. Answers are grounded in the "
        "official *Mapúa Student Handbook* and evaluated for faithfulness by Llama 3.3."
    )

    with st.spinner("Loading AI components..."):
        try:
            vector_store, generator_llm, judge_llm = load_ai_components()
            ai_ready = True
        except Exception as e:
            st.error(f"Could not load AI models: {e}")
            ai_ready = False

    if not ai_ready:
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metrics" in msg:
                st.markdown(msg["metrics"], unsafe_allow_html=True)

    user_query = st.chat_input(
        "E.g., What is the penalty for a 2nd offense of losing a Cardinal Plus ID?"
    )

    if user_query:
        st.chat_message("user").markdown(user_query)

        docs = vector_store.similarity_search(user_query, k=4)
        ctx  = "\n\n".join(d.page_content for d in docs)

        prompt = f"""You are the official AI Assistant for the Mapúa University Student Handbook.

CRITICAL RULE: Answer STRICTLY using ONLY the information in the CONTEXT below.
If the answer is not there, reply exactly:
"I am sorry, but that information is not covered in the Mapúa Student Handbook."

CONTEXT:
{ctx}

USER QUESTION:
{user_query}"""

        with st.spinner("Generating answer..."):
            response = generator_llm.invoke(prompt).content

        FALLBACK = "I am sorry, but that information is not covered in the Mapúa Student Handbook."

        with st.chat_message("assistant"):
            st.markdown(response)

            if FALLBACK in response:
                metrics_text = "📊 **Ragas:** N/A — Successful out-of-scope fallback"
            else:
                try:
                    with st.spinner("Evaluating with Llama 3.3 judge..."):
                        answer_relevancy.strictness = 1
                        dataset = Dataset.from_dict({
                            "question": [user_query],
                            "answer":   [response],
                            "contexts": [[d.page_content for d in docs]],
                        })
                        hf_embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        result   = evaluate(
                            dataset=dataset,
                            metrics=[faithfulness, answer_relevancy],
                            llm=judge_llm,
                            embeddings=hf_embed,
                            run_config=RunConfig(max_workers=1),
                        )
                        rdf = result.to_pandas()
                        f_s = float(rdf["faithfulness"].iloc[0])     if "faithfulness"     in rdf.columns else float("nan")
                        r_s = float(rdf["answer_relevancy"].iloc[0]) if "answer_relevancy" in rdf.columns else float("nan")

                        def score_badge(s: float) -> str:
                            if math.isnan(s):
                                return "N/A"
                            color = "#22c55e" if s >= 0.75 else ("#f97316" if s >= 0.5 else "#ef4444")
                            return (
                                f'<span style="background:{color};color:#fff;'
                                f'padding:1px 8px;border-radius:999px;font-size:12px">'
                                f'{s:.2f}</span>'
                            )

                        metrics_text = (
                            f"📊 **Ragas Evaluation** &nbsp;&nbsp;"
                            f"Faithfulness: {score_badge(f_s)} &nbsp; "
                            f"Relevancy: {score_badge(r_s)}"
                        )
                except Exception as e:
                    metrics_text = f"📊 **Ragas Error:** {e}"

            st.markdown(metrics_text, unsafe_allow_html=True)

        st.session_state.messages += [
            {"role": "user",      "content": user_query},
            {"role": "assistant", "content": response, "metrics": metrics_text},
        ]

    with st.sidebar:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        with st.expander("📚 Retrieved context (last query)"):
            if st.session_state.messages:
                try:
                    last_q = next(
                        m["content"] for m in reversed(st.session_state.messages)
                        if m["role"] == "user"
                    )
                    ctx_docs = vector_store.similarity_search(last_q, k=4)
                    for i, d in enumerate(ctx_docs, 1):
                        st.caption(f"**Chunk {i}** — page {d.metadata.get('page','?')}")
                        st.markdown(d.page_content[:400] + "…")
                        st.divider()
                except StopIteration:
                    st.caption("No queries yet.")
            else:
                st.caption("Ask a question first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — COMPLIANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Compliance Report":
    st.header("📋 Daily Compliance Report Generator")
    st.markdown(
        "Select a date, enter a recipient email, and click *Generate Report* "
        "to produce an AI-written formal report and email draft powered by Cohere Command."
    )

    with st.spinner("Loading AI components..."):
        try:
            vector_store, generator_llm, judge_llm = load_ai_components()
            ai_ready = True
        except Exception as e:
            st.error(f"Could not load AI models: {e}")
            ai_ready = False

    if ai_ready:
        full_df = dr.load_and_prepare()

        if full_df.empty:
            st.warning("No attendance records found in Supabase. Check your connection or tables.")
        else:
            # ── Row 1: Date picker + Email input ─────────────────────────────────
            col_date, col_email = st.columns([1, 2])

            with col_date:
                # Drop nulls/NaT values so python can sort cleanly
                clean_dates = full_df["_date"].dropna().unique()
                available_dates = sorted([d for d in clean_dates if not pd.isna(d)], reverse=True)
                default_date = available_dates[0] if available_dates else date.today()
                selected_date = st.date_input(
                    "Report Date",
                    value=default_date,
                    min_value=min(available_dates) if available_dates else date.today(),
                    max_value=date.today(),
                )

            with col_email:
                recipient_email = st.text_input(
                    "Recipient Email Address",
                    placeholder="discipline.office@mapua.edu.ph",
                )

            # ── Filtered data for selected date ──────────────────────────────────
            daily_df = dr.filter_by_date(full_df, selected_date)
            stats = dr.build_stats(daily_df)

            st.divider()

            # ── Metrics row ───────────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Students Logged", stats["total"])
            m2.metric("Fully Compliant", stats["compliant"])
            m3.metric("Missing ID", stats["lanyard_violations"])
            m4.metric("Dress Code Violations", stats["dress_violations"])

            # ── Filtered table ────────────────────────────────────────────────────
            if daily_df.empty:
                st.info(
                    f"No attendance records found for *{selected_date}*. "
                    "This may be a weekend, holiday, or a date before the system was deployed."
                )
            else:
                # Clean columns to display beautifully
                disp_df = daily_df.copy()
                for col in ["Cloud_Synced", "Lanyard_Compliant", "DressCode_Compliant"]:
                    if col in disp_df.columns:
                        disp_df[col] = disp_df[col].apply(lambda v: "✅" if v else "❌")
                if "Timestamp" in disp_df.columns:
                    disp_df["Timestamp"] = disp_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                # Drop internal _date column
                if "_date" in disp_df.columns:
                    disp_df = disp_df.drop(columns=["_date"])
                st.dataframe(disp_df, use_container_width=True, hide_index=True)

            st.divider()

            # ── Generate Report button ────────────────────────────────────────────
            generate_clicked = st.button("Generate Report", type="primary", disabled=daily_df.empty)

            if generate_clicked:
                if not recipient_email or "@" not in recipient_email:
                    st.error("Please enter a valid recipient email address before generating the report.")
                else:
                    with st.spinner("Retrieving handbook policies and generating report via Cohere..."):
                        try:
                            word_report, email_draft = dr.generate_report(
                                llm=generator_llm,
                                vector_store=vector_store,
                                selected_date=selected_date,
                                recipient_email=recipient_email,
                                daily_df=daily_df,
                            )
                            st.session_state["report_word"] = word_report
                            st.session_state["report_email"] = email_draft
                            st.session_state["report_date"] = selected_date
                        except Exception as e:
                            st.error(f"Report generation failed: {e}")

            # ── Render outputs (persisted across reruns) ──────────────────────────
            if "report_word" in st.session_state and st.session_state.get("report_date") == selected_date:
                report_date = st.session_state["report_date"]

                with st.expander("📄 Part 1 — Formal Administrative Report (Word Document)", expanded=True):
                    edited_report = st.text_area(
                        label="Word Report",
                        value=st.session_state["report_word"],
                        height=400,
                        key="report_word_editor",
                        label_visibility="collapsed",
                    )
                    c_dl, c_spacer, c_sv = st.columns([1.5, 4.5, 2])
                    with c_dl:
                        docx_bytes = dr.export_to_docx(edited_report, report_date)
                        st.download_button(
                            label="⬇️ Download as .docx",
                            data=docx_bytes,
                            file_name=f"compliance_report_{report_date}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
                    with c_sv:
                        if st.button("💾 Save Report Edits", key="save_report_draft", use_container_width=True):
                            st.session_state["report_word"] = edited_report
                            st.success("Report draft changes saved!")

                with st.expander("✉️ Part 2 — Email Draft", expanded=True):
                    edited_email = st.text_area(
                        label="Email Draft",
                        value=st.session_state["report_email"],
                        height=250,
                        key="report_email_editor",
                        label_visibility="collapsed",
                    )
                    c_em_spacer, c_em_sv = st.columns([6, 2])
                    with c_em_sv:
                        if st.button("💾 Save Email Edits", key="save_email_draft", use_container_width=True):
                            st.session_state["report_email"] = edited_email
                            st.success("Email draft changes saved!")

                    st.divider()

                    send_clicked = st.button("Send Email Now", type="primary")
                    if send_clicked:
                        if not recipient_email or "@" not in recipient_email:
                            st.error("Enter a valid recipient email address at the top of the page first.")
                        else:
                            docx_bytes = dr.export_to_docx(edited_report, report_date)
                            filename = f"compliance_report_{report_date}.docx"
                            subject = f"Daily Compliance Report — {report_date}"
                            with st.spinner(f"Sending email to {recipient_email}..."):
                                try:
                                    dr.send_email(
                                        recipient=recipient_email,
                                        subject=subject,
                                        body=edited_email,
                                        docx_bytes=docx_bytes,
                                        docx_filename=filename,
                                    )
                                    st.success(f"Email sent successfully to *{recipient_email}*.")
                                except ValueError as e:
                                    st.error(str(e))
                                    st.info(
                                        "*Setup required:* Add these two lines to your .env file (or Streamlit Secrets):\n\n"
                                        "```\nSMTP_SENDER=your_gmail@gmail.com\n"
                                        "SMTP_PASSWORD=xxxx xxxx xxxx xxxx\n```\n\n"
                                        "The password must be a *Gmail App Password*, not your account password. "
                                        "Generate one at: *Google Account → Security → 2-Step Verification → App Passwords*."
                                    )
                                except Exception as e:
                                    st.error(f"Failed to send email: {e}")