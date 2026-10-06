from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATASET_PATH=Path(__file__).resolve().parent/"sample_projects.csv"
FEATURES=["cost_overrun","fund_utilisation","funds_released_ratio","time_elapsed","progress_vs_time","spend_vs_progress"]

KINDS={
    "Roads":["Highway widening","Bridge repair","Rural link road"],
    "Water":["Pipeline upgrade","Treatment plant","Canal lining"],
    "Schools":["School block","Lab wing","Hostel building"],
    "Health":["Primary health centre","District hospital wing","Trauma unit"],
    "Power":["Substation","Feeder upgrade","Solar park"]
}
DISTRICTS=["Nagpur","Wardha","Amravati","Chandrapur","Bhandara","Yavatmal","Gondia","Akola"]
SECTORS={"Roads":(40,9),"Water":(30,8),"Schools":(18,6),"Health":(26,7),"Power":(55,12)}

def generate(n,seed=7,start=2000,anomaly_rate=.1):
    r=np.random.default_rng(seed)
    rows=[]
    for i in range(n):
        sector=r.choice(list(SECTORS)); base,sd=SECTORS[sector]
        est=max(4,r.normal(base,sd))*1e7
        plan=int(r.integers(240,840)); u=r.uniform(.25,1.25)
        progress=float(np.clip(100*min(u,1.05)+r.normal(0,7),3,100))
        overrun=r.normal(.06,.05); elapsed=int(plan*u)
        released=est*float(np.clip(progress/100+.1+r.normal(0,.04),.05,1))
        utilisation=r.normal(.9,.04)
        if r.random()<anomaly_rate:
            t=int(r.integers(0,5))
            if t==0: overrun=r.uniform(.55,1.45)
            elif t==1: progress=r.uniform(8,22); released=est*r.uniform(.8,.95); utilisation=.97
            elif t==2: elapsed=int(plan*r.uniform(1.5,2)); progress=r.uniform(20,40)
            elif t==3: utilisation=r.uniform(.2,.4)
            else: released=est; progress=r.uniform(15,25); overrun=.3
        rows.append({"id":f"PRJ-{start+i}","name":f"{r.choice(KINDS[sector])} #{i+1}","sector":sector,"district":r.choice(DISTRICTS),"est_cost":est,"actual_cost":est*(1+overrun),"funds_released":released,"funds_used":released*float(np.clip(utilisation,.05,1)),"planned_days":plan,"elapsed_days":elapsed,"progress_pct":progress})
    return pd.DataFrame(rows)

def normalize(data):
    data=data.copy()
    rename={"project_id":"id","estimated_cost":"est_cost","funds_utilised":"funds_used","progress":"progress_pct","time_elapsed":"time_elapsed_ratio"}
    data=data.rename(columns={k:v for k,v in rename.items() if k in data.columns and v not in data.columns})
    defaults={"id":[f"PRJ-{1000+i}" for i in range(len(data))],"name":["Public project"]*len(data),"sector":["Other"]*len(data),"district":["Unknown"]*len(data)}
    for col,vals in defaults.items():
        if col not in data: data[col]=vals
    for col in ["est_cost","actual_cost","funds_released","funds_used","planned_days","elapsed_days","progress_pct"]:
        if col in data: data[col]=pd.to_numeric(data[col],errors="coerce")
    if "planned_days" not in data: data["planned_days"]=100.0
    if "elapsed_days" not in data:
        ratio=pd.to_numeric(data["time_elapsed_ratio"],errors="coerce").fillna(0) if "time_elapsed_ratio" in data else pd.Series(0,index=data.index)
        data["elapsed_days"]=ratio*data["planned_days"]
    if "progress_pct" not in data: data["progress_pct"]=0
    if data["progress_pct"].max(skipna=True)<=1.5: data["progress_pct"]=data["progress_pct"]*100
    for col in ["est_cost","actual_cost","funds_released","funds_used","planned_days","elapsed_days","progress_pct"]:
        data[col]=data[col].replace([np.inf,-np.inf],np.nan).fillna(0)
    data["progress_pct"]=data["progress_pct"].clip(0,100)
    return data

def load_data(target=260):
    data=normalize(pd.read_csv(DATASET_PATH))
    if len(data)<target:
        data=pd.concat([data,generate(target-len(data),seed=19,start=2000)],ignore_index=True)
    return data

def make_features(data):
    features=pd.DataFrame(index=data.index)
    features["cost_overrun"]=(data["actual_cost"]-data["est_cost"])/data["est_cost"].replace(0,np.nan)
    features["fund_utilisation"]=data["funds_used"]/data["funds_released"].replace(0,np.nan)
    features["funds_released_ratio"]=data["funds_released"]/data["est_cost"].replace(0,np.nan)
    features["time_elapsed"]=data["elapsed_days"]/data["planned_days"].replace(0,np.nan)
    features["progress_vs_time"]=data["progress_pct"]/100-np.minimum(1,features["time_elapsed"])
    features["spend_vs_progress"]=features["fund_utilisation"]-data["progress_pct"]/100
    return features.replace([np.inf,-np.inf],np.nan).fillna(0)

def run_anomaly_detection(contamination=.12):
    data=load_data(); features=make_features(data)
    scaler=StandardScaler()
    X=scaler.fit_transform(features[FEATURES])
    model=IsolationForest(n_estimators=250,contamination=contamination,random_state=42)
    data[FEATURES]=features[FEATURES]
    data["anomaly"]=model.fit_predict(X)
    data["anomaly_score"]=model.decision_function(X)
    data["anomaly_strength"]=(-data["anomaly_score"]).clip(lower=0)
    return {"data":data,"model":model,"scaler":scaler,"features":FEATURES}

def explain_project(row,data):
    peers=data[data["sector"]==row["sector"]]
    reasons=[]
    peer_cost=peers["cost_overrun"].median()
    peer_util=peers["fund_utilisation"].median()
    peer_gap=peers["spend_vs_progress"].median()
    if row["cost_overrun"]>peer_cost+.15:
        reasons.append(f"Cost overrun is {row['cost_overrun']:.0%}, above the sector peer median of {peer_cost:.0%}.")
    if row["fund_utilisation"]>peer_util+.15:
        reasons.append(f"Fund utilisation is {row['fund_utilisation']:.0%}, above the sector peer median of {peer_util:.0%}.")
    if row["spend_vs_progress"]>peer_gap+.20:
        reasons.append("Spending is running ahead of reported physical progress.")
    if row["progress_vs_time"]<-.20:
        reasons.append("Physical progress is materially behind the elapsed timeline.")
    if not reasons and row["anomaly"]==-1:
        reasons.append("The combination of project metrics is unusual relative to the overall portfolio.")
    return reasons
