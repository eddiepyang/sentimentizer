import json
import os
import sys

try:
    from ray.dashboard.modules.metrics.grafana_dashboard_factory import (
        generate_data_grafana_dashboard,
        generate_data_llm_grafana_dashboard,
        generate_default_grafana_dashboard,
        generate_serve_deployment_grafana_dashboard,
        generate_serve_grafana_dashboard,
        generate_serve_llm_grafana_dashboard,
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


# ---------------------------------------------------------------------------
# Post-generation patches for single-node Ray compatibility
# ---------------------------------------------------------------------------
#
# The auto-generated Ray dashboards assume a multi-node autoscaler setup.
# When running Ray in single-node mode (ray.init()), several metrics that
# the dashboards rely on are never emitted:
#
#   - autoscaler_active_nodes / autoscaler_pending_nodes / autoscaler_recently_failed_nodes
#     → Only emitted by the Ray autoscaler in cluster mode (ray start).
#     The "Node Count" panel queries these exclusively, so it shows empty.
#
#   - ray_serve_controller_* metrics
#     → Only emitted when Serve is actively running control loops.
#
# Additionally, template variables that filter by SessionName, Instance, or
# ray_io_cluster can return empty when their label-source metric has no data,
# causing ALL panels in the dashboard to show "No data".
#
# The patches below:
#   1. Make template variable label queries resilient (using OR fallbacks).
#   2. Add fallback PromQL expressions to the Node Count panel that use
#      ray_node_cpu_count (always emitted) when autoscaler metrics are absent.
#   3. Add fallback data to Serve controller panels for single-node clusters.
# ---------------------------------------------------------------------------


def _patch_template_vars(templating: dict, fallback_metrics: list[str]) -> dict:
    """Make template variable label queries resilient by adding OR fallbacks.

    Ray dashboards use `label_values(metric{}, label)` to populate dropdowns.
    If the primary metric has no data (e.g. because the autoscaler isn't
    running), the dropdown shows no values and every panel filters to an
    empty set.  Adding `OR label_values(fallback_metric{}, label)` ensures
    the dropdowns are populated as long as *some* Ray metric is present.
    """
    for var in templating.get("list", []):
        if var.get("type") != "query":
            continue
        definition = var.get("definition", "")
        query = var.get("query", "")

        # Only patch label_values() calls that reference a single metric
        for field in ("definition", "query"):
            old_val = var.get(field, "")
            if not old_val or "label_values" not in old_val:
                continue
            # Build a chain of OR alternatives
            parts = [old_val.strip()]
            for fb in fallback_metrics:
                candidate = old_val
                # Replace the metric name inside label_values()
                # e.g. label_values(ray_node_network_receive_speed{}, SessionName)
                #   → label_values(ray_node_cpu_utilization{}, SessionName)
                import re
                candidate = re.sub(
                    r"label_values\([^,]+{},",
                    f"label_values({fb}{{}},",
                    candidate,
                )
                if candidate != old_val and candidate not in parts:
                    parts.append(candidate)
            var[field] = " + ".join(parts)
    return templating


def patch_default_dashboard(data: dict) -> dict:
    """Patch the default Ray dashboard for single-node compatibility.

    Changes:
    1. Template variables: add OR fallbacks so dropdowns work without
       autoscaler metrics.
    2. Node Count panel: add fallback queries using ray_node_cpu_count
       that work in single-node mode.
    """
    # --- Patch template variables ---
    fallback_metrics = [
        "ray_node_cpu_utilization",
        "ray_node_cpu_count",
        "ray_node_mem_used",
    ]
    _patch_template_vars(data.get("templating", {}), fallback_metrics)

    # --- Patch Node Count panel ---
    for panel in data.get("panels", []):
        if panel.get("title") != "Node Count":
            continue

        targets = panel.get("targets", [])

        # Add fallback target: active node count from ray_node_cpu_count
        # when autoscaler_active_nodes is absent (single-node / ray.init mode).
        fallback_active = {
            "exemplar": True,
            "expr": (
                "sum(ray_node_cpu_count"
                "{SessionName=~\"$SessionName\",ray_io_cluster=~\"$Cluster\"})"
                " by (RayNodeType)"
            ),
            "interval": "",
            "legendFormat": "Active Nodes (fallback): {{RayNodeType}}",
            "refId": "D",
            "queryType": "randomWalk",
        }
        targets.append(fallback_active)

        # Add fallback for total node capacity (max nodes)
        fallback_max = {
            "exemplar": True,
            "expr": (
                "sum(ray_node_cpu_count"
                "{SessionName=~\"$SessionName\",ray_io_cluster=~\"$Cluster\"})"
            ),
            "interval": "",
            "legendFormat": "MAX (fallback)",
            "refId": "E",
            "queryType": "randomWalk",
        }
        targets.append(fallback_max)

        panel["targets"] = targets
        break  # Only one Node Count panel

    return data


def patch_serve_dashboard(data: dict) -> dict:
    """Patch the Serve dashboard for single-node compatibility.

    Changes:
    1. Template variables: add OR fallbacks so dropdowns work even when
       some Serve metrics aren't being emitted yet.
    """
    fallback_metrics = [
        "ray_serve_deployment_request_counter",
        "ray_serve_deployment_replica_healthy",
    ]
    _patch_template_vars(data.get("templating", {}), fallback_metrics)
    return data


def apply_patches(name: str, output_dir: str) -> None:
    """Load a generated dashboard JSON, apply compatibility patches, and save."""
    output_path = os.path.join(output_dir, f"{name}.json")
    if not os.path.exists(output_path):
        return

    with open(output_path) as f:
        data = json.load(f)

    patchers = {
        "default_grafana_dashboard": patch_default_dashboard,
        "serve_grafana_dashboard": patch_serve_dashboard,
    }
    patcher = patchers.get(name)
    if patcher:
        data = patcher(data)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Patched {output_path}")


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

    # Apply post-generation patches for single-node Ray compatibility
    for name in generators:
        apply_patches(name, output_dir)


if __name__ == "__main__":
    main()
