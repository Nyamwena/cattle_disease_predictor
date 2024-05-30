from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict_general/', views.predict_general, name='predict_general'),
    path('predict_theileriosis/', views.predict_theileriosis, name='predict_theileriosis'),
    path('history/', views.prediction_history, name='prediction_history'),
    path('plot_predictions_by_location/', views.plot_predictions_by_location, name='plot_predictions_by_location'),
    path('plot_predictions_by_season/', views.plot_predictions_by_season, name='plot_predictions_by_season'),
    path('plot_heatmap/', views.plot_heatmap, name='plot_heatmap'),
    path('plot_predictions_over_time/', views.plot_predictions_over_time, name='plot_predictions_over_time'),
    path('plot_predictions_by_disease/', views.plot_predictions_by_disease, name='plot_predictions_by_disease'),
    path('plot_scatter_location_season/', views.plot_scatter_location_season, name='plot_scatter_location_season'),
    path('login/', auth_views.LoginView.as_view(template_name='predictor/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='index'), name='logout'),
]
