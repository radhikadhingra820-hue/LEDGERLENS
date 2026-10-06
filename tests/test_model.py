from model import load_data, run_anomaly_detection, explain_project

def test_dataset_loads():
    data=load_data()
    assert len(data)>=100
    assert {"id","sector","progress_pct"}.issubset(data.columns)

def test_anomaly_detection_runs():
    result=run_anomaly_detection()
    data=result["data"]
    assert {"anomaly","anomaly_score","fund_utilisation"}.issubset(data.columns)
    assert len(data)>=100

def test_project_explanation_returns_list():
    result=run_anomaly_detection()
    row=result["data"].iloc[0]
    assert isinstance(explain_project(row,result["data"]),list)
