from model import load_data, run_anomaly_detection, explain_project


def test_dataset_loads():
    data=load_data()
    assert len(data)>0
    assert "sector" in data.columns
    assert "progress" in data.columns


def test_anomaly_detection_runs():
    result=run_anomaly_detection()
    data=result["data"]
    assert "anomaly" in data.columns
    assert "anomaly_score" in data.columns
    assert len(data)>0


def test_project_explanation_returns_list():
    result=run_anomaly_detection()
    row=result["data"].iloc[0]
    reasons=explain_project(row,result["data"])
    assert isinstance(reasons,list)
