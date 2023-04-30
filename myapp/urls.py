from django.urls import path
from . import views

urlpatterns = [
    path("predict/temperature", views.predict_temperature, name="predict/temperature"),path("predict/humidity", views.predict_humidity, name="predict/humidity"),
    path("predict/light", views.predict_light, name="predict/light"),path("predict/co2", views.predict_co2, name="predict/co2"),
    path("predict/occupancy", views.predict_occupancy, name="predict/occupancy"),
]
