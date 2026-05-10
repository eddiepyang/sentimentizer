import json
import os
import re
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


def _fix_promql_text(text: str) -> str:
    """Fix common PromQL and label_values() formatting issues in dashboard text.

    Applies the following fixes:
    1. Remove empty braces after metric names: metric{} → metric
       (Grafana rejects label_values(metric{}, label) and query_result(func(metric{})).
    2. Remove trailing commas inside label selectors: {foo="bar",} → {foo="bar"}
       (PromQL does not allow trailing commas in label matchers.)
    3. Replace single-quoted string matchers with double quotes: resource='GPU' → resource="GPU"
       (PromQL requires double-quoted strings; single quotes are invalid.)
    4. Remove spaces before =~ and !~ operators: foo =~"bar" → foo=~"bar"
       (While some parsers tolerate the space, it's non-standard and can cause issues.)
    """
    # 1. Remove empty braces after metric names (but NOT inside legendFormat or other
    #    Grafana template syntax which uses {{ }}).
    #    Match patterns like: metric_name{}, metric_name{ } but NOT {{ variable }}
    text = re.sub(r"(\w+)\{\s*\}", r"\1", text)

    # 2. Remove trailing commas inside label selectors: {foo="bar",} → {foo="bar"}
    text = re.sub(r",\s*\}", "}", text)

    # 3. Replace single-quoted string matchers with double quotes in PromQL
    #    e.g. resource='GPU' → resource="GPU"
    text = re.sub(r"([a-zA-Z_]\w*)='([^']*)'", r'\1="\2"', text)

    # 4. Remove spaces before =~ and !~ operators
    text = re.sub(r"([a-zA-Z_]\w*)\s*(=~|!~)", r"\1\2", text)

    return text


def _patch_template_vars(templating: dict, fallback_metrics: list[str]) -> dict:
    """Make template variable label queries resilient by adding OR fallbacks.

    Ray dashboards use `label_values(metric{}, label)` to populate dropdowns.
    If the primary metric has no data (e.g. because the autoscaler isn't
    running), the dropdown shows no values and every panel filters to an
    empty set.  Adding `OR label_values(fallback_metric{}, label)` ensures
    the dropdowns are populated as long as *some* Ray metric is present.

    This function also fixes formatting issues in the generated expressions:
    - Strips empty {} after metric names (e.g. label_values(metric{}, label) → label_values(metric, label))
    - Removes trailing commas inside label selectors
    - Fixes single-quoted strings and spacing issues
    """
    for var in templating.get("list", []):
        if var.get("type") != "query":
            continue

        # Determine which fields to patch. The query field can be either a
        # plain string or a dict with a "query" sub-key (Grafana v8+ format).
        fields_to_patch = {}
        definition = var.get("definition", "")
        if definition and "label_values" in definition:
            fields_to_patch["definition"] = definition

        query_val = var.get("query", "")
        if isinstance(query_val, str) and query_val and "label_values" in query_val:
            fields_to_patch["query"] = query_val
        elif isinstance(query_val, dict):
            # Grafana v8+ uses {"query": "...", "refId": "..."} format
            inner_query = query_val.get("query", "")
            if inner_query and "label_values" in inner_query:
                fields_to_patch["query"] = inner_query
                fields_to_patch["_query_is_dict"] = True

        for field, old_val in fields_to_patch.items():
            if field == "_query_is_dict":
                continue
            if not old_val:
                continue

            # First, fix any formatting issues in the original value
            fixed_val = _fix_promql_text(old_val)

            # Build a chain of OR fallback alternatives using label_values()
            parts = [fixed_val.strip()]
            for fb in fallback_metrics:
                # Replace the metric name (and its optional label selector) inside
                # label_values() with the fallback metric name, preserving the label
                # selector if present.
                #
                # Correctly handles both forms:
                #   label_values(metric_name, label) → label_values(fb, label)
                #   label_values(metric_name{filters}, label) → label_values(fb{filters}, label)
                #
                # The regex matches the metric name with optional {filters} and
                # replaces only the metric name portion, keeping any filters intact.
                candidate = re.sub(
                    r"label_values\((\w+)\{([^}]*)\},",
                    lambda m: f"label_values({fb}{{{m.group(2)}}},",
                    fixed_val,
                )
                # Also handle the simple case without braces: label_values(metric, label)
                candidate = re.sub(
                    r"label_values\((\w+),",
                    f"label_values({fb},",
                    candidate,
                    count=1,  # Only replace the first match to avoid double-replacing
                )
                if candidate != fixed_val and candidate not in parts:
                    parts.append(candidate)

            new_val = " + ".join(parts)

            if field == "query" and fields_to_patch.get("_query_is_dict"):
                var["query"]["query"] = new_val
            else:
                var[field] = new_val

    return templating


def _fix_panel_targets(data: dict) -> dict:
    """Fix PromQL expressions in all panel targets recursively.

    Handles both top-level panels and nested panels (rows containing sub-panels).
    Applies _fix_promql_text to each expr value.
    """
    def _fix_panels(panels):
        for panel in panels:
            for target in panel.get("targets", []):
                if "expr" in target:
                    target["expr"] = _fix_promql_text(target["expr"])
            # Recurse into row panels that contain sub-panels
            _fix_panels(panel.get("panels", []))

    _fix_panels(data.get("panels", []))
    return data


def patch_default_dashboard(data: dict) -> dict:
    """Patch the default Ray dashboard for single-node compatibility.

    Changes:
    1. Template variables: add OR fallbacks so dropdowns work without
       autoscaler metrics.
    2. Node Count panel: add fallback queries using ray_node_cpu_count
       that work in single-node mode.
    3. Fix any PromQL formatting issues in panel targets.
    """
    # --- Patch template variables ---
    fallback_metrics = [
        "ray_node_cpu_utilization",
        "ray_node_cpu_count",
        "ray_node_mem_used",
    ]
    _patch_template_vars(data.get("templating", {}), fallback_metrics)

    # --- Fix PromQL in panel targets ---
    _fix_panel_targets(data)

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
                '{SessionName=~"$SessionName",ray_io_cluster=~"$Cluster"})'
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
                '{SessionName=~"$SessionName",ray_io_cluster=~"$Cluster"})'
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
    2. Fix PromQL formatting issues (single quotes, spacing) in panel targets.
    """
    fallback_metrics = [
        "ray_serve_deployment_request_counter",
        "ray_serve_deployment_replica_healthy",
    ]
    _patch_template_vars(data.get("templating", {}), fallback_metrics)

    # Fix PromQL in panel targets (single quotes, spacing, etc.)
    _fix_panel_targets(data)

    return data


def patch_train_dashboard(data: dict) -> dict:
    """Patch the Train dashboard for single-node compatibility.

    Changes:
    1. Fix malformed PromQL label selectors generated by empty global filters
       (e.g., trailing commas or double braces).
    2. Fix template variable label_values expressions.
    """
    # Fix template variables (empty braces, trailing commas)
    _patch_template_vars(data.get("templating", {}), [])

    # Fix PromQL in panel targets (double braces, trailing commas, etc.)
    _fix_panel_targets(data)

    return data


def apply_patches(name: str, output_dir: str) -> None:
    """Load a generated dashboard JSON, apply compatibility patches, and save.

    Applies the following global text-level fixes before JSON parsing:
    1. Remove {{{global_filters}}} Jinja2-style placeholders → empty string.
    2. Replace double braces {{ }} → single braces { } (PromQL label selectors).
    3. Fix trailing commas inside label selectors: {foo="bar",} → {foo="bar"}.
    4. Remove empty braces after metric names: metric{} → metric.
    5. Replace single-quoted string matchers: resource='GPU' → resource="GPU".
    6. Remove spaces before =~ and !~ operators.

    Then applies per-dashboard structural patches (template variable fallbacks,
    Node Count fallback targets, etc.).
    """
    output_path = os.path.join(output_dir, f"{name}.json")
    if not os.path.exists(output_path):
        return

    with open(output_path) as f:
        content = f.read()

    # --- Text-level fixes (applied before JSON parsing) ---

    # Remove Jinja2-style {{{global_filters}}} / {{global_filters}} placeholders.
    # Ray's generator uses these but they don't resolve to anything in our context.
    content = content.replace("{{{global_filters}}}", "")
    content = content.replace("{{global_filters}}", "")

    # Fix double braces left over from Python string formatting:
    # e.g. {{SessionName}} → {SessionName}  (valid PromQL label matcher)
    # but preserve Grafana template variables like {{model_type}} in legendFormat
    # by only fixing double braces that contain PromQL-like content (=~ or = or !~).
    # We do this by replacing {{ only when followed by a label-matcher pattern,
    # and }} only when preceded by one.
    content = re.sub(r"\{\{(\w+[=~!])", r"{\1", content)
    content = re.sub(r"([\"'][=~!])\}\}", r"\1}", content)
    # Also handle cases like {{variable}} that are NOT PromQL — leave those alone.

    # Fix Prometheus parse error: unexpected left brace '{'
    # Ray's dashboard generator produces `label_values(metric{}, label)` which
    # is rejected by modern Grafana versions. We strip the empty braces.
    content = re.sub(r"label_values\(([^,]+?)\{\}\s*,", r"label_values(\1,", content)

    # Also strip empty {} in other contexts (e.g. query_result(func(metric{}))).
    # But be careful not to strip {{ }} Grafana template syntax.
    # Match: word char followed by {}  but NOT {{ }}
    content = re.sub(r"(?<!\{)(\w+)\{\s*\}(?!\})", r"\1", content)

    # Fix trailing commas inside label selectors: {job="ray",} → {job="ray"}
    content = re.sub(r",\s*\}", "}", content)

    # Fix single-quoted string matchers in PromQL: resource='GPU' → resource="GPU"
    content = re.sub(r"(\w+)='([^']*)'", r'\1="\2"', content)

    # Fix spaces before =~ and !~ operators: foo =~"bar" → foo=~"bar"
    content = re.sub(r"(\w+)\s*(=~|!~)", r"\1\2", content)

    # --- JSON-level structural patches ---
    data = json.loads(content)

    patchers = {
        "default_grafana_dashboard": patch_default_dashboard,
        "serve_grafana_dashboard": patch_serve_dashboard,
        "train_grafana_dashboard": patch_train_dashboard,
    }
    patcher = patchers.get(name)
    if patcher:
        data = patcher(data)

    # For ALL dashboards: fix template variables and panel targets that
    # weren't covered by a specific patcher.
    if name not in patchers:
        # Still fix PromQL formatting issues in template vars and panels
        _patch_template_vars(data.get("templating", {}), [])
        _fix_panel_targets(data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Patched {output_path}")


def generate_ml_metrics_dashboard():
    """Generate a custom dashboard for Sentimentizer ML metrics."""
    data = {
        "__inputs": [
            {
                "name": "DS_PROMETHEUS",
                "label": "Prometheus",
                "description": "",
                "type": "datasource",
                "pluginId": "prometheus",
                "pluginName": "Prometheus",
            }
        ],
        "__requires": [
            {"type": "grafana", "id": "grafana", "name": "Grafana", "version": "9.0.0"},
            {
                "type": "datasource",
                "id": "prometheus",
                "name": "Prometheus",
                "version": "1.0.0",
            },
            {
                "type": "panel",
                "id": "graph",
                "name": "Graph (Old)",
                "version": "",
            },
            {
                "type": "panel",
                "id": "table",
                "name": "Table",
                "version": "",
            },
        ],
        "annotations": {
            "list": [
                {
                    "builtIn": 1,
                    "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                    "enable": True,
                    "hide": True,
                    "iconColor": "rgba(0, 211, 255, 1)",
                    "name": "Annotations & Alerts",
                    "type": "dashboard",
                }
            ]
        },
        "title": "Sentimentizer ML Metrics",
        "uid": "sentimentizerMLMetrics",
        "version": 1,
        "schemaVersion": 27,
        "style": "dark",
        "refresh": "5s",
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {
            "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "1h"],
        },
        "templating": {
            "list": [
                {
                    "current": {},
                    "hide": 0,
                    "includeAll": False,
                    "label": "Datasource",
                    "multi": False,
                    "name": "datasource",
                    "options": [],
                    "query": "prometheus",
                    "queryValue": "",
                    "refresh": 1,
                    "regex": "",
                    "skipUrlSync": False,
                    "type": "datasource",
                },
                {
                    "allValue": ".*",
                    "current": {"selected": True, "text": "All", "value": "$__all"},
                    "datasource": "${datasource}",
                    "definition": "label_values(sentimentizer_training_train_loss, model_type)",
                    "hide": 0,
                    "includeAll": True,
                    "label": "Model Type",
                    "multi": True,
                    "name": "model_type",
                    "options": [],
                    "query": {
                        "qryType": 1,
                        "query": "label_values(sentimentizer_training_train_loss, model_type)",
                        "refId": "VariableQuery",
                    },
                    "refresh": 2,
                    "regex": "",
                    "skipUrlSync": False,
                    "sort": 1,
                    "type": "query",
                },
            ]
        },
        "panels": [
            {
                "title": "Loss (Train vs Val)",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "${datasource}"},
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [
                    {
                        "expr": 'sentimentizer_training_train_loss{model_type=~"$model_type"}',
                        "legendFormat": "Train Loss ({{model_type}})",
                        "refId": "A",
                    },
                    {
                        "expr": 'sentimentizer_training_val_loss{model_type=~"$model_type"}',
                        "legendFormat": "Val Loss ({{model_type}})",
                        "refId": "B",
                    },
                ],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineWidth": 2},
                        "unit": "none",
                    },
                    "overrides": [],
                },
            },
            {
                "title": "Validation Core Metrics",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "${datasource}"},
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": 'sentimentizer_training_val_accuracy{model_type=~"$model_type"}',
                        "legendFormat": "Accuracy ({{model_type}})",
                        "refId": "A",
                    },
                    {
                        "expr": 'sentimentizer_training_val_f1{model_type=~"$model_type"}',
                        "legendFormat": "F1 Score ({{model_type}})",
                        "refId": "B",
                    },
                    {
                        "expr": 'sentimentizer_training_val_precision{model_type=~"$model_type"}',
                        "legendFormat": "Precision ({{model_type}})",
                        "refId": "C",
                    },
                    {
                        "expr": 'sentimentizer_training_val_recall{model_type=~"$model_type"}',
                        "legendFormat": "Recall ({{model_type}})",
                        "refId": "D",
                    },
                ],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineWidth": 2},
                        "unit": "none",
                        "min": 0,
                        "max": 1,
                    },
                    "overrides": [],
                },
            },
            {
                "title": "Current Epoch & Metrics Table",
                "type": "table",
                "datasource": {"type": "prometheus", "uid": "${datasource}"},
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
                "targets": [
                    {
                        "expr": 'sentimentizer_training_epoch{model_type=~"$model_type"}',
                        "format": "table",
                        "instant": True,
                        "legendFormat": "Epoch",
                        "refId": "A",
                    },
                    {
                        "expr": 'sentimentizer_training_train_loss{model_type=~"$model_type"}',
                        "format": "table",
                        "instant": True,
                        "legendFormat": "Train Loss",
                        "refId": "B",
                    },
                    {
                        "expr": 'sentimentizer_training_val_loss{model_type=~"$model_type"}',
                        "format": "table",
                        "instant": True,
                        "legendFormat": "Val Loss",
                        "refId": "C",
                    },
                    {
                        "expr": 'sentimentizer_training_val_cohen_kappa{model_type=~"$model_type"}',
                        "format": "table",
                        "instant": True,
                        "legendFormat": "Kappa",
                        "refId": "D",
                    },
                    {
                        "expr": 'sentimentizer_training_val_precision{model_type=~"$model_type"}',
                        "format": "table",
                        "instant": True,
                        "legendFormat": "Precision",
                        "refId": "E",
                    },
                    {
                        "expr": 'sentimentizer_training_val_recall{model_type=~"$model_type"}',
                        "format": "table",
                        "instant": True,
                        "legendFormat": "Recall",
                        "refId": "F",
                    },
                    {
                        "expr": 'sentimentizer_training_val_f1{model_type=~"$model_type"}',
                        "format": "table",
                        "instant": True,
                        "legendFormat": "F1",
                        "refId": "G",
                    },
                    {
                        "expr": 'sentimentizer_training_val_positive_accuracy{model_type=~"$model_type"}',
                        "format": "table",
                        "instant": True,
                        "legendFormat": "Pos Acc",
                        "refId": "H",
                    },
                    {
                        "expr": 'sentimentizer_training_val_negative_accuracy{model_type=~"$model_type"}',
                        "format": "table",
                        "instant": True,
                        "legendFormat": "Neg Acc",
                        "refId": "I",
                    },
                ],
                "transformations": [
                    {"id": "merge", "options": {}},
                    {
                        "id": "organize",
                        "options": {
                            "excludeByName": {
                                "Time": False,
                                "__name__": True,
                                "instance": True,
                                "job": True,
                            },
                            "renameByName": {
                                "Value #A": "Epoch",
                                "Value #B": "Train Loss",
                                "Value #C": "Val Loss",
                                "Value #D": "Kappa",
                                "Value #E": "Precision",
                                "Value #F": "Recall",
                                "Value #G": "F1",
                                "Value #H": "Pos Acc",
                                "Value #I": "Neg Acc",
                            },
                        },
                    },
                ],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"align": "auto", "displayMode": "auto"},
                        "decimals": 4,
                    }
                },
            },
        ],
    }
    return json.dumps(data, indent=4), None


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
        "ml_metrics_dashboard": generate_ml_metrics_dashboard,
    }

    for name, generator in generators.items():
        save_dashboard(name, generator, output_dir)

    # Apply post-generation patches for single-node Ray compatibility
    for name in generators:
        apply_patches(name, output_dir)


if __name__ == "__main__":
    main()