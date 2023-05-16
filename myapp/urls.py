from django.urls import path
from . import views

urlpatterns = [
    path("predict/temperature", views.predict_temperature, name="predict/temperature"),path("predict/power", views.predict_power, name="predict/power"),
    path("predict/current", views.predict_current, name="predict/current"),
    path("predict/voltage", views.predict_voltage, name="predict/voltage"),
]
