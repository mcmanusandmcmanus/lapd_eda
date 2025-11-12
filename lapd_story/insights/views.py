import json

from django.utils.safestring import mark_safe
from django.views.generic import TemplateView

from . import services


class DashboardView(TemplateView):
    template_name = "insights/dashboard.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["summary_cards"] = services.build_summary_cards()
        priority_mix = services.build_priority_mix()
        priority_rows = [
            {"label": label, "share": share, "display": f"{share:.1%}"}
            for label, share in priority_mix.items()
        ]
        context["priority_mix_items"] = sorted(
            priority_rows, key=lambda item: item["share"], reverse=True
        )
        context["call_type_rows"] = services.top_call_types()
        context["spotlights"] = services.build_story_spotlights()
        context["dataset_profile"] = services.dataset_profile()
        context["categorical_rollup"] = services.categorical_rollup()

        daily_chart = services.build_daily_chart()
        hourly_chart = services.build_hourly_chart()
        heatmap = services.build_area_heatmap()
        monthly_priority = services.build_monthly_priority_chart()
        family_trend = services.build_family_trend_chart()

        context["daily_chart_json"] = mark_safe(json.dumps(daily_chart)) if daily_chart else "{}"
        context["hourly_chart_json"] = mark_safe(json.dumps(hourly_chart)) if hourly_chart else "{}"
        context["monthly_priority_chart_json"] = (
            mark_safe(json.dumps(monthly_priority)) if monthly_priority else "{}"
        )
        context["family_trend_chart_json"] = (
            mark_safe(json.dumps(family_trend)) if family_trend else "{}"
        )
        context["heatmap_hours"] = heatmap.get("hours", [])

        matrix = heatmap.get("matrix", [])
        areas = heatmap.get("areas", [])
        peak_value = max((max(row) for row in matrix), default=0)
        normalized_rows = []
        for area, values in zip(areas, matrix):
            cell_data = [
                {
                    "value": int(value),
                    "intensity": min(float(value / peak_value), 1.0) if peak_value else 0.0,
                }
                for value in values
            ]
            normalized_rows.append((area, cell_data))
        context["heatmap_rows"] = normalized_rows
        return context


class LabView(TemplateView):
    template_name = "insights/lab.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["lab_headlines"] = services.lab_headlines()
        context["lab_metadata"] = services.lab_metadata()
        context["lab_feature_importances"] = services.lab_feature_importances()
        context["lab_feature_catalog"] = services.lab_feature_catalog()
        context["lab_confusion_rows"] = services.lab_confusion_rows()
        context["lab_confusion_labels"] = services.lab_confusion_labels()
        context["lab_classification_rows"] = services.lab_classification_rows()
        context["lab_notes"] = services.lab_notes()

        feature_chart = services.lab_feature_chart()
        context["feature_chart_json"] = (
            mark_safe(json.dumps(feature_chart)) if feature_chart else "{}"
        )
        return context
