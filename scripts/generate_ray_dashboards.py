import os
import json
import sys

try:
    from ray.dashboard.modules.metrics.grafana_dashboard_factory import (
        generate_default_grafana_dashboard,
        generate_serve_grafana_dashboard,
        generate_serve_deployment_grafana_dashboard,
        generate_serve_llm_grafana_dashboard,
        generate_data_grafana_dashboard,
        generate_data_llm_grafana_dashboard,
        generate_train_grafana_dashboard,
    )
except ImportError as e:
    print(f"Error: Could not import Ray dashboard factory: {e}")
    print("Make sure ray[default] or ray[dashboard] is installed.")
    sys.exit(1)

def save_dashboard(name, generator, output_dir):
    try:
        content, _ = generator()
        output_path = os.path.join(output_dir, f"{name}.json")
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Generated {output_path}")
    except Exception as e:
        print(f"Error generating {name}: {e}")

def main():
    output_dir = "metrics/grafana/dashboards"
    os.makedirs(output_dir, exist_ok=True)
    
    generators = {
        "default_grafana_dashboard": generate_default_grafana_dashboard,
        "serve_grafana_dashboard": generate_serve_grafana_dashboard,
        "serve_deployment_grafana_dashboard": generate_serve_deployment_grafana_dashboard,
        "serve_llm_grafana_dashboard": generate_serve_llm_grafana_dashboard,
        "data_grafana_dashboard": generate_data_grafana_dashboard,
        "data_llm_grafana_dashboard": generate_data_llm_grafana_dashboard,
        "train_grafana_dashboard": generate_train_grafana_dashboard,
    }
    
    for name, generator in generators.items():
        save_dashboard(name, generator, output_dir)

if __name__ == "__main__":
    main()
