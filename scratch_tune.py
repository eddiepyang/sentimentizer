import json


def generate_tune_metrics_dashboard():
    _DS = {"type": "prometheus", "uid": "prometheus"}

    def _target(metric: str, legend: str, ref: str) -> dict:
        lbl = '{model_type=~"$model_type"}'
        expr = f"{metric}{lbl}"
        return {"datasource": _DS, "expr": expr, "legendFormat": legend, "refId": ref}

    def _target_trial(metric: str, legend: str, ref: str) -> dict:
        # Per-trial metrics have trial_id
        lbl = '{model_type=~"$model_type", trial_id=~".*"}'
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
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
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
                "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
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
                "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
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
                "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                "targets": [_target("sentimentizer_tune_best_val_f1", "Best F1", "A")],
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
                }
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
