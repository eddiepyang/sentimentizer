import json

with open("metrics/grafana/dashboards/ml_metrics_dashboard.json") as f:
    d = json.load(f)
for p in d["panels"]:
    if p["type"] == "table":
        print(json.dumps(p["transformations"], indent=2))
