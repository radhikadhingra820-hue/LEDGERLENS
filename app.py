import streamlit as st
import pandas as pd
from model import run_anomaly_detection, explain_project

st.set_page_config(page_title="LEDGERLENS",page_icon="🔎",layout="wide")
st.markdown("""
<style>
.stApp{background:#0b0f19}
.block-container{max-width:1200px;padding-top:2rem;padding-bottom:3rem}
.hero{padding:1.6rem 1.8rem;border:1px solid #26344b;border-radius:18px;background:linear-gradient(135deg,#111827,#182235);margin-bottom:1.2rem}
.hero-title{font-size:2.5rem;font-weight:800;letter-spacing:.05em}
.hero-subtitle{color:#aab7cb;font-size:1rem;margin-top:.35rem}
</style>
""",unsafe_allow_html=True)
st.markdown("""
<div class="hero"><div class="hero-title">🔎 LEDGERLENS</div><div class="hero-subtitle">Public Development Project Anomaly Detection</div></div>
""",unsafe_allow_html=True)

@st.cache_data
def get_results():
    return run_anomaly_detection()

results=get_results()
data=results["data"]
a,b,c,d=st.columns(4)
a.metric("Projects",f"{len(data):,}")
b.metric("Anomalies",f"{int((data['anomaly']==-1).sum()):,}")
c.metric("Avg Utilisation",f"{data['fund_utilisation'].mean():.1%}")
d.metric("Model","Isolation Forest")
st.divider()

overview,review,details=st.tabs(["📊 Overview","🚨 Review Queue","🔍 Project Details"])
with overview:
    st.subheader("Project Overview")
    left,right=st.columns(2)
    with left:
        st.markdown("### Sector Distribution")
        st.bar_chart(data["sector"].value_counts())
    with right:
        st.markdown("### Anomaly Score Distribution")
        hist=pd.cut(data["anomaly_score"],bins=10).value_counts().sort_index()
        hist.index=hist.index.astype(str)
        st.bar_chart(hist)
    st.markdown("### Spending vs Progress")
    chart_data=data.assign(fund_utilisation_pct=data["fund_utilisation"]*100)
    st.scatter_chart(chart_data,x="fund_utilisation_pct",y="progress_pct",color="sector")

with review:
    st.subheader("Projects Flagged for Review")
    flagged=data[data["anomaly"]==-1].sort_values("anomaly_score").copy()
    st.caption("Lower anomaly scores indicate stronger model-level evidence of unusual behaviour.")
    cols=["id","name","sector","district","est_cost","actual_cost","fund_utilisation","progress_pct","anomaly_score"]
    st.dataframe(flagged[cols].rename(columns={"id":"Project","name":"Project Name","est_cost":"Estimated Cost","actual_cost":"Actual Cost","fund_utilisation":"Fund Utilisation","progress_pct":"Progress","anomaly_score":"Anomaly Score"}),use_container_width=True)

with details:
    st.subheader("Project Details")
    project_id=st.selectbox("Select project",data["id"].tolist())
    row=data[data["id"]==project_id].iloc[0]
    st.write(f"### {row['name']}")
    st.write(f"**Sector:** {row['sector']}  ·  **District:** {row['district']}")
    cols=st.columns(4)
    cols[0].metric("Estimated Cost",f"₹{row['est_cost']:,.0f}")
    cols[1].metric("Actual Cost",f"₹{row['actual_cost']:,.0f}")
    cols[2].metric("Fund Utilisation",f"{row['fund_utilisation']:.1%}")
    cols[3].metric("Progress",f"{row['progress_pct']:.1f}%")
    if row["anomaly"]==-1:
        st.error("Flagged for review by the baseline detector.")
    else:
        st.success("Not flagged by the baseline detector.")
    st.markdown("### Why was it flagged?")
    reasons=explain_project(row,data)
    if reasons:
        for reason in reasons: st.warning(reason)
    else:
        st.info("No strong rule-based signal was found for this project.")
    st.markdown("### Project Metrics")
    metric_cols=["est_cost","actual_cost","funds_released","funds_used","planned_days","elapsed_days","progress_pct","cost_overrun","fund_utilisation","funds_released_ratio","time_elapsed","progress_vs_time","spend_vs_progress"]
    st.dataframe(row[metric_cols].to_frame("Value"),use_container_width=True)
