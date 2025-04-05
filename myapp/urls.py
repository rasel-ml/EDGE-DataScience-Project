from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analysis/', views.analysis, name='analysis'),
    path('prediction/', views.prediction, name='prediction'),
    path('update/', views.update, name='update'),
    path('predict_weather/', views.predict_weather, name='predict_weather'),
    path('visualization/', views.visualization, name='visualization'),
    path('generate_graphs/', views.generate_graphs, name='generate_graphs'),
    path('process_weather_analysis/', views.process_weather_analysis, name='process_weather_analysis'),
    path('predict_weather/', views.predict_weather, name='predict_weather'),
    ]