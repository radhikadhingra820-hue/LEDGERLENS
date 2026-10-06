from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATASET_PATH=Path(__file__).resolve().parent/"sample_projects.csv"

FEATURES=[
    "cost_overrun","fund_utilisation","funds_released_ratio",
    "time_elapsed","progress_vs_time","spend_vs_progress"
]

def load_data():
    data=pd.read_csv(DATASET_PATH)
    data["estimated_cost"]=pd.to_numeric(data["estimated_cost"],errors="coerce").fillna(0)
    data["actual_cost"]=pd.to_numeric(data["actual_cost"],errors="coerce").fillna(0)
    data["funds_released"]=pd.to_numeric(data["funds_released"],errors="coerce").fillna(0)
    data["funds_utilised"]=pd.to_numeric(data["funds_utilised"],errors="coerce").fillna(0)
    data["progress"]=pd.to_numeric(data["progress"],errors="coerce").clip(0,1).fillna(0)
    data["time_elapsed"]=pd.to_numeric(data["time_elapsed"],errors="coerce").clip(0,1).fillna(0)

    data["cost_overrun"]=(data["actual_cost"]-data["estimated_cost"])/data["estimated_cost"].replace(0,np.nan)
    data["cost_overrun"]=data["cost_overrun"].replace([np.inf,-np.inf],0).fillna(0)
    data["fund_utilisation"]=data["funds_utilised"]/data["funds_released"].replace(0,np.nan)
    data["fund_utilisation"]=data["fund_utilisation"].replace([np.inf,-np.inf],0).fillna(0).clip(0,1)
    data["funds_released_ratio"]=data["funds_released"]/data["estimated_cost"].replace(0,np.nan)
    data["funds_released_ratio"]=data["funds_released_ratio"].replace([np.inf,-np.inf],0).fillna(0)
    data["progress_vs_time"]=data["progress"]-data["time_elapsed"]
    data["spend_vs_progress"]=data["fund_utilisation"]-data["progress"]
    return data

def run_anomaly_detection(contamination=0.12):
    data=load_data()
    scaler=StandardScaler()
    X=scaler.fit_transform(data[FEATURES])
    model=IsolationForest(n_estimators=250,contamination=contamination,random_state=42)
    data["anomaly"]=model.fit_predict(X)
    data["anomaly_score"]=model.decision_function(X)
    return {"data":data,"model":model,"scaler":scaler,"features":FEATURES}

def explain_project(row,data):
    peers=data[data["sector"]==row["sector"]]
    reasons=[]
    peer_cost=peers["cost_overrun"].median()
    peer_util=peers["fund_utilisation"].median()
    peer_gap=peers["spend_vs_progress"].median()

    if row["cost_overrun"]>peer_cost+0.15:
        reasons.append("Cost overrun is noticeably higher than the sector peer median.")
    if row["fund_utilisation"]>peer_util+0.15:
        reasons.append("Fund utilisation is unusually high compared with similar projects.")
    if row["spend_vs_progress"]>peer_gap+0.20:
        reasons.append("Spending is running ahead of reported project progress.")
    if row["progress_vs_time"]<-0.20:
        reasons.append("Physical progress is lagging behind the elapsed timeline.")
    return reasons
