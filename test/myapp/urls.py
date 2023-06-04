from django.urls import path
from . import views

urlpatterns = [
<<<<<<< HEAD
    path("predict/temperature", views.predict_temperature, name="predict/temperature"),path("predict/power", views.predict_power, name="predict/power"),
    path("predict/current", views.predict_current, name="predict/current"),
    path("predict/voltage", views.predict_voltage, name="predict/voltage"),
    path("predict/network", views.predict_network, name="predict/network"),
=======
    path("predict/temperature", views.predict_temperature, name="predict/temperature"),path("predict/humidity", views.predict_humidity, name="predict/humidity"),
    path("predict/light", views.predict_light, name="predict/light"),path("predict/co2", views.predict_co2, name="predict/co2"),
    path("predict/occupancy", views.predict_occupancy, name="predict/occupancy"),
>>>>>>> parent of 9969882 (code refactoring)
]
