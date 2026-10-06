import streamlit as st
import pandas as pd
from model import run_anomaly_detection, explain_project, load_data

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
<div class="hero">
<div class="hero-title">🔎 LEDGERLENS</div>
<div class="hero-subtitle">Public Development Project Anomaly Detection</div>
</div>
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
overview,review,insights=st.tabs(["📊 Overview","🚨 Review Queue","🔍 Project Details"])

with overview:
    st.subheader("Project Overview")
    left,right=st.columns(2)
    with left:
        st.markdown("### Sector Distribution")
        st.bar_chart(data["sector"].value_counts())
    with right:
        st.markdown("### Anomaly Score Distribution")
        st.bar_chart(data["anomaly_score"].round(2).value_counts().sort_index())
    st.markdown("### Spending vs Progress")
    st.scatter_chart(data,x="fund_utilisation",y="progress",color="sector")

with review:
    st.subheader("Projects Flagged for Review")
    flagged=data[data["anomaly"]==-1].sort_values("anomaly_score").copy()
    st.caption("Lower anomaly scores indicate stronger model-level evidence of unusual behaviour.")
    st.dataframe(flagged[["project_id","sector","estimated_cost","actual_cost","fund_utilisation","progress","anomaly_score"]],use_container_width=True)

with insights:
    st.subheader("Project Details")
    project_id=st.selectbox("Select project",data["project_id"].tolist())
    row=data[data["project_id"]==project_id].iloc[0]
    st.write(f"### {row['project_id']}")
    st.write(f"**Sector:** {row['sector']}")
    cols=st.columns(4)
    cols[0].metric("Estimated Cost",f"₹{row['estimated_cost']:,.0f}")
    cols[1].metric("Actual Cost",f"₹{row['actual_cost']:,.0f}")
    cols[2].metric("Fund Utilisation",f"{row['fund_utilisation']:.1%}")
    cols[3].metric("Progress",f"{row['progress']:.1%}")
    reasons=explain_project(row,data)
    st.markdown("### Why was it flagged?")
    if reasons:
        for reason in reasons:
            st.warning(reason)
    else:
        st.success("No major anomaly signals were identified by the explanation rules.")
    st.markdown("### Peer Comparison")
    st.dataframe(row[["sector","estimated_cost","actual_cost","funds_released","fund_utilisation","progress","time_elapsed"]].to_frame("Value"),use_container_width=True)
