import json

def fix_dash(path):
    with open(path, 'r') as f:
        dash = json.load(f)
    
    modified = False
    for var in dash.get('templating', {}).get('list', []):
        if var['name'] == 'ModelType':
            if 'datasource' not in var or var['datasource'] != '${datasource}':
                var['datasource'] = '${datasource}'
                modified = True
            
            # Make sure query is properly formatted if needed, though string is usually fine
            if isinstance(var['query'], str):
                var['query'] = {"query": var['query'], "refId": "StandardVariableQuery"}
                modified = True

    if modified:
        with open(path, 'w') as f:
            json.dump(dash, f, indent=2)
        print(f"Fixed {path}")

fix_dash('metrics/grafana/dashboards/tune_grafana_dashboard.json')
fix_dash('metrics/grafana/dashboards/train_grafana_dashboard.json')
