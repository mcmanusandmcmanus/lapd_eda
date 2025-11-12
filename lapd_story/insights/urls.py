from django.urls import path

from .views import DashboardView, LabView

app_name = "insights"

urlpatterns = [
    path("", DashboardView.as_view(), name="dashboard"),
    path("lab/", LabView.as_view(), name="lab"),
]
