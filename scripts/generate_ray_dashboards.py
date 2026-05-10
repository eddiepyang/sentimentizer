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


from collections.abc import Callable
from typing import Any


def save_dashboard(name: str, generator: Callable[[], tuple[str, Any]], output_dir: str) -> None:
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


def _fix_promql_expr(expr: str) -> str:
    """Fix common PromQL formatting issues in a single expression string.

    This operates on parsed JSON string values (not raw JSON text), so it's
    safe to replace quotes and other characters that would break JSON syntax.

    Applies the following fixes:
    1. Remove empty braces after metric names: metric{} → metric
    2. Remove trailing commas inside label selectors: {foo="bar",} → {foo="bar"}
    3. Replace single-quoted string matchers with double quotes: resource='GPU' → resource="GPU"
    4. Remove spaces before =~ and !~ operators: foo =~"bar" → foo=~"bar"
    5. Convert double braces {{ }} to single braces { } in PromQL label matchers
       (e.g. {{SessionName=~"$SessionName"}} → {SessionName=~"$SessionName"})
       while preserving Grafana template variables like {{model_type}}.
    """
    # 5. Fix double braces that are PromQL label matchers (not Grafana template vars).
    #    PromQL: {{label=~"value"}} → {label=~"value"}
    #    Grafana template: {{variable}} → keep as-is
    #    Strategy: {{ followed by \w+= or \w+!~ or \w+=~ is a PromQL matcher start.
    #    }} preceded by " is a PromQL matcher end.
    expr = re.sub(r"\{\{(\w+[=~!])", r"{\1", expr)
    expr = re.sub(r'"\}\}', '"}', expr)

    # 1. Remove empty braces after metric names: metric{} → metric
    #    But do NOT touch {{variable}} Grafana template syntax.
    expr = re.sub(r"(\w+)\{\s*\}", r"\1", expr)

    # 2. Remove trailing commas inside label selectors: {foo="bar",} → {foo="bar"}
    expr = re.sub(r",\s*\}", "}", expr)

    # 2b. Remove leading commas inside label selectors: {, foo="bar"} → {foo="bar"}
    #     This happens when {{{global_filters}}} is removed from expressions like
    #     {{{global_filters}}}, deployment=~"$deployment"}
    expr = re.sub(r"\{\s*,\s*", "{", expr)

    # 2c. Fix multiple consecutive commas: {foo="bar", , baz="qux"} → {foo="bar", baz="qux"}
    expr = re.sub(r",\s*,", ",", expr)

    # 3. Replace single-quoted string matchers with double quotes in PromQL
    #    e.g. resource='GPU' → resource="GPU"
    #    This is safe here because we're operating on parsed JSON string values,
    #    not raw JSON text. The double quotes don't need JSON escaping at this level
    #    because json.dump() will handle the escaping when writing the file.
    expr = re.sub(r"([a-zA-Z_]\w*)='([^']*)'", r'\1="\2"', expr)

    # 4. Remove spaces before =~ and !~ operators
    expr = re.sub(r"([a-zA-Z_]\w*)\s*(=~|!~)", r"\1\2", expr)

    return expr


def _patch_template_vars(templating: dict, fallback_metrics: list[str]) -> dict:
    """Make template variable label queries resilient by adding OR fallbacks.

    Ray dashboards use `label_values(metric{}, label)` to populate dropdowns.
    If the primary metric has no data (e.g. because the autoscaler isn't
    running), the dropdown shows no values and every panel filters to an
    empty set.  Adding `OR label_values(fallback_metric{}, label)` ensures
    the dropdowns are populated as long as *some* Ray metric is present.

    This function also fixes formatting issues in the generated expressions:
    - Strips empty {} after metric names
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
            fixed_val = _fix_promql_expr(old_val)
            new_val = fixed_val

            # Replace the metric name inside label_values() with the first highly reliable
            # fallback metric (like ray_node_cpu_count) so the dropdown always populates.
            # Grafana does NOT support combining label_values() calls with `+` or `OR`.
            if fallback_metrics:
                fb = fallback_metrics[0]
                # Replace metric with {filters}
                candidate = re.sub(
                    r"label_values\((\w+)\{([^}]*)\},",
                    lambda m, _fb=fb: f"label_values({_fb}{{{m.group(2)}}},",
                    fixed_val,
                )
                # Replace metric without braces
                candidate = re.sub(
                    r"label_values\((\w+),",
                    f"label_values({fb},",
                    candidate,
                    count=1,
                )
                new_val = candidate

            if field == "query" and fields_to_patch.get("_query_is_dict"):
                var["query"]["query"] = new_val
            else:
                var[field] = new_val

    return templating


def _fix_panel_targets(data: dict) -> dict:
    """Fix PromQL expressions in all panel targets recursively.

    Handles both top-level panels and nested panels (rows containing sub-panels).
    Applies _fix_promql_expr to each expr value, but NOT to legendFormat or
    other Grafana template syntax fields.
    """

    def _fix_panels(panels: list[dict]) -> None:
        for panel in panels:
            for target in panel.get("targets", []):
                if "expr" in target:
                    target["expr"] = _fix_promql_expr(target["expr"])
            # Recurse into row panels that contain sub-panels
            _fix_panels(panel.get("panels", []))

    _fix_panels(data.get("panels", []))
    return data


def _fix_templating_exprs(templating: dict) -> dict:
    """Fix PromQL expressions in template variable definitions (no fallbacks).

    Applies _fix_promql_expr to definition and query fields of query-type
    template variables. This is used for dashboards that don't need fallback
    metrics but still need formatting fixes.
    """
    for var in templating.get("list", []):
        if var.get("type") != "query":
            continue

        definition = var.get("definition", "")
        if definition:
            var["definition"] = _fix_promql_expr(definition)

        query_val = var.get("query", "")
        if isinstance(query_val, str) and query_val:
            var["query"] = _fix_promql_expr(query_val)
        elif isinstance(query_val, dict):
            inner_query = query_val.get("query", "")
            if inner_query:
                query_val["query"] = _fix_promql_expr(inner_query)

    return templating


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
                "sum(ray_node_cpu_count" '{SessionName=~"$SessionName",ray_io_cluster=~"$Cluster"})'
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

    Applies the following text-level fixes BEFORE JSON parsing (safe operations
    that only modify JSON string content, not JSON structure):
    1. Remove {{{global_filters}}} Jinja2-style placeholders → empty string.
    2. Remove empty braces after metric names in label_values() calls.
    3. Remove trailing commas inside label selectors: {foo="bar",} → {foo="bar"}.

    Then applies JSON-level patches AFTER parsing (where we can safely modify
    PromQL expressions inside parsed string values without breaking JSON):
    4. Fix double braces {{ }} in PromQL label matchers → single braces.
    5. Replace single-quoted string matchers: resource='GPU' → resource="GPU".
    6. Remove spaces before =~ and !~ operators.
    7. Per-dashboard structural patches (template variable fallbacks, etc.).
    """
    output_path = os.path.join(output_dir, f"{name}.json")
    if not os.path.exists(output_path):
        return

    with open(output_path) as f:
        content = f.read()

    # --- Text-level fixes (applied before JSON parsing) ---
    # These are safe because they only modify content inside JSON strings
    # without changing the JSON structure itself (no quote conflicts).

    # Remove Jinja2-style {{{global_filters}}} / {{global_filters}} placeholders.
    # Ray's generator uses these but they don't resolve to anything in our context.
    content = content.replace("{{{global_filters}}}", "")
    content = content.replace("{{global_filters}}", "")

    # Fix Prometheus parse error: unexpected left brace '{'
    # Ray's dashboard generator produces `label_values(metric{}, label)` which
    # is rejected by modern Grafana versions. We strip the empty braces.
    content = re.sub(r"label_values\(([^,]+?)\{\}\s*,", r"label_values(\1,", content)

    # Also strip empty {} in other PromQL contexts inside JSON strings.
    # e.g. query_result(func(metric{})) → query_result(func(metric))
    # Be careful not to strip {{ }} Grafana template syntax (used in legendFormat).
    # Match: word char followed by {} but NOT preceded by { and NOT followed by }
    content = re.sub(r"(?<!\{)(\w+)\{\s*\}(?!\})", r"\1", content)

    # Fix trailing commas inside label selectors: {job="ray",} → {job="ray"}
    content = re.sub(r",\s*\}", "}", content)

    # Fix leading commas inside label selectors: {, foo="bar"} → {foo="bar"}
    # This happens when {{{global_filters}}} is removed from expressions like
    # metric{{{global_filters}}}, deployment=~"$deployment"}
    content = re.sub(r"\{\s*,\s*", "{", content)

    # Fix multiple consecutive commas: {foo="bar", , baz="qux"} → {foo="bar", baz="qux"}
    # This happens when {{{global_filters}}} is sandwiched between other labels
    content = re.sub(r",\s*,", ",", content)

    # --- JSON-level structural patches ---
    data = json.loads(content)

    # Strip __inputs and __requires — these are Grafana *import* metadata fields
    # that are NOT used (and cause warnings/errors) when loading dashboards via
    # file provisioning.  Grafana logs a JSON parsing error for provisioned
    # dashboards that still contain these keys.
    data.pop("__inputs", None)
    data.pop("__requires", None)

    # Rewrite any datasource references that still use the placeholder UID
    # "DS_PROMETHEUS" (emitted by Ray's generator for some dashboards) to the
    # stable uid "prometheus" that matches our provisioned datasource.
    _rewrite_datasource_uid(data, old_uid="DS_PROMETHEUS", new_uid="prometheus")

    patchers = {
        "default_grafana_dashboard": patch_default_dashboard,
        "serve_grafana_dashboard": patch_serve_dashboard,
        "train_grafana_dashboard": patch_train_dashboard,
    }
    patcher = patchers.get(name)
    if patcher:
        data = patcher(data)

    # For ALL dashboards: fix PromQL formatting issues in template vars and
    # panel targets. This handles single quotes, spacing, double braces, and
    # empty {} that weren't caught by text-level fixes.
    if name not in patchers:
        _fix_templating_exprs(data.get("templating", {}))
        _fix_panel_targets(data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Patched {output_path}")


def _rewrite_datasource_uid(data: dict, old_uid: str, new_uid: str) -> None:
    """Recursively rewrite datasource UID references in panels and annotations.

    Rewrites both string datasources (legacy format) and dict datasources
    ({"type": ..., "uid": old_uid}) in panels, rows, and annotation lists.
    """

    def _fix_panels(panels: list[dict]) -> None:
        for panel in panels:
            ds = panel.get("datasource")
            if ds == old_uid:
                panel["datasource"] = {"type": "prometheus", "uid": new_uid}
            elif isinstance(ds, dict) and ds.get("uid") == old_uid:
                ds["uid"] = new_uid
            _fix_panels(panel.get("panels", []))
            for target in panel.get("targets", []):
                t_ds = target.get("datasource")
                if t_ds == old_uid:
                    target["datasource"] = {"type": "prometheus", "uid": new_uid}
                elif isinstance(t_ds, dict) and t_ds.get("uid") == old_uid:
                    t_ds["uid"] = new_uid

    _fix_panels(data.get("panels", []))

    for ann in data.get("annotations", {}).get("list", []):
        ann_ds = ann.get("datasource")
        if ann_ds == old_uid:
            ann["datasource"] = {"type": "prometheus", "uid": new_uid}
        elif isinstance(ann_ds, dict) and ann_ds.get("uid") == old_uid:
            ann_ds["uid"] = new_uid


def generate_ml_metrics_dashboard() -> tuple[str, None]:
    """Generate the Sentimentizer Training dashboard.

    Dashboard UID: ``sentimentizerTraining`` (docs/metrics.md line 150).

    Panel queries use PromQL ``or`` to fall back between the two metric sources
    described in docs/metrics.md lines 158-163:

    - ``sentimentizer_training_*`` (port 8081, standalone exporter) — available
      after each training epoch and persisted to disk after training completes.
    - ``ray_sentimentizer_live_*`` (port 8080, Ray workers) — only available
      while Ray is running; Ray adds the ``ray_`` prefix automatically.

    This ensures the dashboard shows data in all three states:
    1. During distributed training (live Ray gauges from rank-0 worker).
    2. During single-node training (exporter gauges from trainer.evaluate()).
    3. After training completes (persisted exporter gauges from JSON file).
    """
    _DS = {"type": "prometheus", "uid": "prometheus"}

    def _target(metric: str, live_metric: str, legend: str, ref: str, **extra: Any) -> dict:
        lbl = '{model_type=~"$model_type"}'
        expr = f"{metric}{lbl}\n  or\n{live_metric}{lbl}"
        return {"datasource": _DS, "expr": expr, "legendFormat": legend, "refId": ref, **extra}

    def _table_target(metric: str, live_metric: str, legend: str, ref: str) -> dict:
        lbl = '{model_type=~"$model_type"}'
        expr = f"{metric}{lbl}\n  or\n{live_metric}{lbl}"
        return {
            "datasource": _DS,
            "expr": expr,
            "format": "table",
            "instant": True,
            "legendFormat": legend,
            "refId": ref,
        }

    data = {
        # __requires stripped by apply_patches(); no __inputs needed.
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
        "title": "Sentimentizer Training",
        "uid": "sentimentizerTraining",
        "version": 1,
        "schemaVersion": 36,
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
                    # Query both families so dropdown populates during distributed
                    # training (Ray live metrics) and after (exporter metrics).
                    "allValue": ".*",
                    "current": {"selected": True, "text": "All", "value": "$__all"},
                    "datasource": _DS,
                    "definition": ("label_values(sentimentizer_training_train_loss, model_type)"),
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
                "title": "Current Epoch",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 4, "x": 0, "y": 0},
                "targets": [
                    _target(
                        "sentimentizer_training_epoch",
                        "ray_sentimentizer_live_epoch",
                        "Epoch ({{model_type}})",
                        "A",
                    )
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "fixed", "fixedColor": "super-light-blue"},
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [{"color": "blue", "value": None}],
                        },
                        "unit": "none",
                    },
                    "overrides": [],
                },
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "orientation": "auto",
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "textMode": "auto",
                },
            },
            {
                "title": "Loss (Train vs Val)",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 10, "x": 4, "y": 0},
                "targets": [
                    _target(
                        "sentimentizer_training_train_loss",
                        "ray_sentimentizer_live_train_loss",
                        "Train Loss ({{model_type}})",
                        "A",
                    ),
                    _target(
                        "sentimentizer_training_val_loss",
                        "ray_sentimentizer_live_val_loss",
                        "Val Loss ({{model_type}})",
                        "B",
                    ),
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
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 10, "x": 14, "y": 0},
                "targets": [
                    _target(
                        "sentimentizer_training_val_accuracy",
                        "ray_sentimentizer_live_val_accuracy",
                        "Accuracy ({{model_type}})",
                        "A",
                    ),
                    _target(
                        "sentimentizer_training_val_f1",
                        "ray_sentimentizer_live_val_f1",
                        "F1 Score ({{model_type}})",
                        "B",
                    ),
                    _target(
                        "sentimentizer_training_val_precision",
                        "ray_sentimentizer_live_val_precision",
                        "Precision ({{model_type}})",
                        "C",
                    ),
                    _target(
                        "sentimentizer_training_val_recall",
                        "ray_sentimentizer_live_val_recall",
                        "Recall ({{model_type}})",
                        "D",
                    ),
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
                "title": "Per-Class Accuracy",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "targets": [
                    _target(
                        "sentimentizer_training_val_positive_accuracy",
                        "ray_sentimentizer_live_val_positive_accuracy",
                        "Positive Acc ({{model_type}})",
                        "A",
                    ),
                    _target(
                        "sentimentizer_training_val_negative_accuracy",
                        "ray_sentimentizer_live_val_negative_accuracy",
                        "Negative Acc ({{model_type}})",
                        "B",
                    ),
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
                "title": "Cohen's Kappa & AUC-ROC",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [
                    _target(
                        "sentimentizer_training_val_cohen_kappa",
                        "ray_sentimentizer_live_val_cohen_kappa",
                        "Cohen's Kappa ({{model_type}})",
                        "A",
                    ),
                    _target(
                        "sentimentizer_training_val_auc_roc",
                        "ray_sentimentizer_live_val_auc_roc",
                        "AUC-ROC ({{model_type}})",
                        "B",
                    ),
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
                "title": "Current Epoch & Metrics Snapshot",
                "type": "table",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
                "targets": [
                    _table_target(
                        "sentimentizer_training_epoch", "ray_sentimentizer_live_epoch", "Epoch", "A"
                    ),
                    _table_target(
                        "sentimentizer_training_train_loss",
                        "ray_sentimentizer_live_train_loss",
                        "Train Loss",
                        "B",
                    ),
                    _table_target(
                        "sentimentizer_training_val_loss",
                        "ray_sentimentizer_live_val_loss",
                        "Val Loss",
                        "C",
                    ),
                    _table_target(
                        "sentimentizer_training_val_accuracy",
                        "ray_sentimentizer_live_val_accuracy",
                        "Accuracy",
                        "D",
                    ),
                    _table_target(
                        "sentimentizer_training_val_f1", "ray_sentimentizer_live_val_f1", "F1", "E"
                    ),
                    _table_target(
                        "sentimentizer_training_val_precision",
                        "ray_sentimentizer_live_val_precision",
                        "Precision",
                        "F",
                    ),
                    _table_target(
                        "sentimentizer_training_val_recall",
                        "ray_sentimentizer_live_val_recall",
                        "Recall",
                        "G",
                    ),
                    _table_target(
                        "sentimentizer_training_val_cohen_kappa",
                        "ray_sentimentizer_live_val_cohen_kappa",
                        "Kappa",
                        "H",
                    ),
                    _table_target(
                        "sentimentizer_training_val_positive_accuracy",
                        "ray_sentimentizer_live_val_positive_accuracy",
                        "Pos Acc",
                        "I",
                    ),
                    _table_target(
                        "sentimentizer_training_val_negative_accuracy",
                        "ray_sentimentizer_live_val_negative_accuracy",
                        "Neg Acc",
                        "J",
                    ),
                    _table_target(
                        "sentimentizer_training_val_auc_roc",
                        "ray_sentimentizer_live_val_auc_roc",
                        "AUC-ROC",
                        "K",
                    ),
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
                                "Value #D": "Accuracy",
                                "Value #E": "F1",
                                "Value #F": "Precision",
                                "Value #G": "Recall",
                                "Value #H": "Kappa",
                                "Value #I": "Pos Acc",
                                "Value #J": "Neg Acc",
                                "Value #K": "AUC-ROC",
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


def generate_tune_metrics_dashboard() -> tuple[str, None]:
    """Generate the Sentimentizer Tuning dashboard.

    Dashboard UID: ``sentimentizerTuning``.

    Covers aggregate trial stats (best metrics, trial counts) and per-trial
    time-series for all metrics emitted during Ray Tune runs.
    """
    _DS = {"type": "prometheus", "uid": "prometheus"}

    def _target(metric: str, legend: str, ref: str) -> dict:
        lbl = '{model_type=~"$model_type"}'
        expr = f"{metric}{lbl}"
        return {"datasource": _DS, "expr": expr, "legendFormat": legend, "refId": ref}

    def _target_trial(metric: str, legend: str, ref: str) -> dict:
        lbl = '{model_type=~"$model_type", trial_id=~"$trial_id"}'
        expr = f"{metric}{lbl}"
        return {"datasource": _DS, "expr": expr, "legendFormat": legend, "refId": ref}

    data = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": [
            {
                "title": "Completed Trials",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 5, "x": 0, "y": 0},
                "targets": [_target("sentimentizer_tune_trial_completed_count", "Completed", "A")],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "Best Val Accuracy",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 5, "x": 5, "y": 0},
                "targets": [_target("sentimentizer_tune_best_val_accuracy", "Best Acc", "A")],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "Best Val Loss",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 5, "x": 10, "y": 0},
                "targets": [_target("sentimentizer_tune_best_val_loss", "Best Loss", "A")],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "Best Val F1",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 4, "x": 15, "y": 0},
                "targets": [_target("sentimentizer_tune_best_val_f1", "Best F1", "A")],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "Trial Count",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 5, "x": 19, "y": 0},
                "targets": [_target("sentimentizer_tune_trial_count", "Trials", "A")],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "Val Accuracy (per trial)",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "targets": [
                    _target_trial("sentimentizer_tune_val_accuracy", "Trial {{trial_id}}", "A")
                ],
            },
            {
                "title": "Train vs Val Loss (per trial)",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [
                    _target_trial("sentimentizer_tune_train_loss", "Train {{trial_id}}", "A"),
                    _target_trial("sentimentizer_tune_val_loss", "Val {{trial_id}}", "B"),
                ],
            },
            {
                "title": "Val Precision (per trial)",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                "targets": [
                    _target_trial("sentimentizer_tune_val_precision", "Trial {{trial_id}}", "A")
                ],
            },
            {
                "title": "Val Recall (per trial)",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                "targets": [
                    _target_trial("sentimentizer_tune_val_recall", "Trial {{trial_id}}", "A")
                ],
            },
            {
                "title": "Val F1 (per trial)",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
                "targets": [_target_trial("sentimentizer_tune_val_f1", "Trial {{trial_id}}", "A")],
            },
            {
                "title": "Val Cohen's Kappa (per trial)",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
                "targets": [
                    _target_trial("sentimentizer_tune_val_cohen_kappa", "Trial {{trial_id}}", "A")
                ],
            },
            {
                "title": "Per-Class Accuracy (per trial)",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 32},
                "targets": [
                    _target_trial(
                        "sentimentizer_tune_val_positive_accuracy", "Pos {{trial_id}}", "A"
                    ),
                    _target_trial(
                        "sentimentizer_tune_val_negative_accuracy", "Neg {{trial_id}}", "B"
                    ),
                ],
            },
            {
                "title": "Epoch (per trial)",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 32},
                "targets": [_target_trial("sentimentizer_tune_epoch", "Trial {{trial_id}}", "A")],
            },
        ],
        "refresh": "5s",
        "schemaVersion": 38,
        "style": "dark",
        "tags": ["sentimentizer", "tune"],
        "templating": {
            "list": [
                {
                    "current": {"selected": False, "text": "All", "value": "$__all"},
                    "datasource": _DS,
                    "definition": "label_values(sentimentizer_tune_trial_count, model_type)",
                    "hide": 0,
                    "includeAll": True,
                    "multi": True,
                    "name": "model_type",
                    "options": [],
                    "query": {
                        "query": "label_values(sentimentizer_tune_trial_count, model_type)",
                        "refId": "StandardVariableQuery",
                    },
                    "refresh": 1,
                    "type": "query",
                },
                {
                    "allValue": ".*",
                    "current": {"selected": False, "text": "All", "value": "$__all"},
                    "datasource": _DS,
                    "definition": "label_values(sentimentizer_tune_val_accuracy, trial_id)",
                    "hide": 0,
                    "includeAll": True,
                    "label": "Trial ID",
                    "multi": True,
                    "name": "trial_id",
                    "options": [],
                    "query": {
                        "query": "label_values(sentimentizer_tune_val_accuracy, trial_id)",
                        "refId": "StandardVariableQuery",
                    },
                    "refresh": 1,
                    "type": "query",
                },
            ]
        },
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {},
        "timezone": "",
        "title": "Sentimentizer Tuning",
        "uid": "sentimentizerTuning",
        "version": 1,
    }
    return json.dumps(data, indent=4), None


def generate_system_metrics_dashboard() -> tuple[str, None]:
    """Generate the Sentimentizer System dashboard.

    Dashboard UID: ``sentimentizerSystem``.

    Visualizes system stats (CPU, memory, disk), GPU metrics, and Ray health
    from the standalone exporter (port 8081).
    """
    _DS = {"type": "prometheus", "uid": "prometheus"}

    def _target(metric: str, legend: str, ref: str, **extra: Any) -> dict:
        return {"datasource": _DS, "expr": metric, "legendFormat": legend, "refId": ref, **extra}

    def _target_gpu(metric: str, legend: str, ref: str) -> dict:
        lbl = '{gpu_index=~"$gpu_index"}'
        expr = f"{metric}{lbl}"
        return {"datasource": _DS, "expr": expr, "legendFormat": legend, "refId": ref}

    data = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": [
            {
                "title": "System Info",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
                "targets": [
                    _target(
                        "sentimentizer_system_info",
                        "{{platform}} / {{python}} / {{cpu_count}} CPUs",
                        "A",
                    )
                ],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "Ray Available",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 4, "w": 3, "x": 6, "y": 0},
                "targets": [_target("sentimentizer_ray_available", "Up", "A")],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "Ray Node Count",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 4, "w": 3, "x": 9, "y": 0},
                "targets": [_target("sentimentizer_ray_node_count", "Nodes", "A")],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "Ray Metric Count",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 4, "w": 3, "x": 12, "y": 0},
                "targets": [_target("sentimentizer_ray_metric_count", "Metrics", "A")],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "Controller State",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 4, "w": 3, "x": 15, "y": 0},
                "targets": [_target("sentimentizer_ray_controller_state", "State", "A")],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "Controller Op Time",
                "type": "stat",
                "datasource": _DS,
                "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0},
                "targets": [
                    _target("sentimentizer_ray_controller_operation_time_s", "Seconds", "A")
                ],
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
            },
            {
                "title": "CPU %",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 4},
                "targets": [_target("sentimentizer_system_cpu_percent", "CPU %", "A")],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineWidth": 2},
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                    },
                    "overrides": [],
                },
            },
            {
                "title": "Memory %",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 4},
                "targets": [_target("sentimentizer_system_memory_percent", "Memory %", "A")],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineWidth": 2},
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                    },
                    "overrides": [],
                },
            },
            {
                "title": "Disk %",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": 4},
                "targets": [_target("sentimentizer_system_disk_percent", "Disk %", "A")],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineWidth": 2},
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                    },
                    "overrides": [],
                },
            },
            {
                "title": "Memory Bytes",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
                "targets": [
                    _target("sentimentizer_system_memory_available_bytes", "Available", "A"),
                    _target("sentimentizer_system_memory_total_bytes", "Total", "B"),
                ],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineWidth": 2},
                        "unit": "bytes",
                    },
                    "overrides": [],
                },
            },
            {
                "title": "Disk Bytes",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
                "targets": [
                    _target("sentimentizer_system_disk_free_bytes", "Free", "A"),
                    _target("sentimentizer_system_disk_total_bytes", "Total", "B"),
                ],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineWidth": 2},
                        "unit": "bytes",
                    },
                    "overrides": [],
                },
            },
            {
                "title": "GPU Utilization",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 20},
                "targets": [
                    _target_gpu("sentimentizer_gpu_utilization_percent", "GPU {{gpu_index}}", "A")
                ],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineWidth": 2},
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                    },
                    "overrides": [],
                },
            },
            {
                "title": "GPU Memory",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 20},
                "targets": [
                    _target_gpu("sentimentizer_gpu_memory_used_bytes", "Used {{gpu_index}}", "A"),
                    _target_gpu("sentimentizer_gpu_memory_total_bytes", "Total {{gpu_index}}", "B"),
                ],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineWidth": 2},
                        "unit": "bytes",
                    },
                    "overrides": [],
                },
            },
            {
                "title": "GPU Temperature",
                "type": "timeseries",
                "datasource": _DS,
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 28},
                "targets": [
                    _target_gpu("sentimentizer_gpu_temperature_celsius", "GPU {{gpu_index}}", "A")
                ],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineWidth": 2},
                        "unit": "celsius",
                        "min": 0,
                    },
                    "overrides": [],
                },
            },
        ],
        "refresh": "5s",
        "schemaVersion": 38,
        "style": "dark",
        "tags": ["sentimentizer", "system"],
        "templating": {
            "list": [
                {
                    "current": {"selected": False, "text": "All", "value": "$__all"},
                    "datasource": _DS,
                    "definition": "label_values(sentimentizer_gpu_utilization_percent, gpu_index)",
                    "hide": 0,
                    "includeAll": True,
                    "label": "GPU Index",
                    "multi": True,
                    "name": "gpu_index",
                    "options": [],
                    "query": {
                        "query": "label_values(sentimentizer_gpu_utilization_percent, gpu_index)",
                        "refId": "StandardVariableQuery",
                    },
                    "refresh": 1,
                    "type": "query",
                }
            ]
        },
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {},
        "timezone": "",
        "title": "Sentimentizer System",
        "uid": "sentimentizerSystem",
        "version": 1,
    }
    return json.dumps(data, indent=4), None


def _cleanup_stale_base_files(output_dir: str) -> None:
    """Remove *_base.json files that cause duplicate UID errors in Grafana.

    These files are artefacts of previous script versions that wrote an
    unpatched copy alongside the patched one.  Grafana loads every .json file
    in the provisioning directory; if two files share the same ``uid`` field
    Grafana logs:

        "the same UID is used more than once"
        "dashboards provisioning provider has no database write permissions
         because of duplicates"

    …and refuses to load ANY dashboard in the folder.  Removing the stale
    ``*_base.json`` files eliminates the duplicates.
    """
    import glob

    removed = []
    for path in glob.glob(os.path.join(output_dir, "*_base.json")):
        os.remove(path)
        removed.append(os.path.basename(path))
    if removed:
        print(f"Removed {len(removed)} stale base file(s): {', '.join(removed)}")


def main() -> None:
    output_dir = "metrics/grafana/dashboards"
    os.makedirs(output_dir, exist_ok=True)

    # Remove *_base.json files that create duplicate UIDs and block Grafana
    # from loading any provisioned dashboard.
    _cleanup_stale_base_files(output_dir)

    generators = {
        "default_grafana_dashboard": generate_default_grafana_dashboard,
        "serve_grafana_dashboard": generate_serve_grafana_dashboard,
        "serve_deployment_grafana_dashboard": generate_serve_deployment_grafana_dashboard,
        "serve_llm_grafana_dashboard": generate_serve_llm_grafana_dashboard,
        "data_grafana_dashboard": generate_data_grafana_dashboard,
        "data_llm_grafana_dashboard": generate_data_llm_grafana_dashboard,
        "train_grafana_dashboard": generate_train_grafana_dashboard,
        "ml_metrics_dashboard": generate_ml_metrics_dashboard,
        "tune_metrics_dashboard": generate_tune_metrics_dashboard,
        "system_metrics_dashboard": generate_system_metrics_dashboard,
    }

    for name, generator in generators.items():
        save_dashboard(name, generator, output_dir)

    # Apply post-generation patches for single-node Ray compatibility
    for name in generators:
        apply_patches(name, output_dir)


if __name__ == "__main__":
    main()
