import json

with open('metrics/grafana/dashboards/train_grafana_dashboard.json', 'r') as f:
    train_dash = json.load(f)

with open('metrics/grafana/dashboards/tune_grafana_dashboard.json', 'r') as f:
    tune_dash = json.load(f)

train_vars = {v['name'] for v in train_dash.get('templating', {}).get('list', [])}
print(f"Train vars: {train_vars}")

for v in tune_dash.get('templating', {}).get('list', []):
    if v['name'] not in train_vars:
        print(f"Appending variable {v['name']}")
        train_dash['templating']['list'].append(v)

with open('metrics/grafana/dashboards/train_grafana_dashboard.json', 'w') as f:
    json.dump(train_dash, f, indent=2)

