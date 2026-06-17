from django.urls import path
from django.contrib.auth import views as auth_views

from .views.views import *
from .views.analysis_views import *
from .views.ml_views import *
from .views.predict_veiws import *

urlpatterns = [
    path("register/", register_view, name="register"),

    path(
        "login/",
        CustomLoginView.as_view(),
        name="login"
    ),

    path(
        "logout/",
        auth_views.LogoutView.as_view(),
        name="logout"
    ),

    path("upload/", upload_dataset, name="upload"),

    path("datasets/", datasets_view, name="datasets"),

    path("datasets/<int:pk>/delete/", delete_dataset, name="delete_dataset"),

    path("datasets/<int:pk>/", dataset_detail_view, name="dataset_detail"),

    path("generate/", generate_dataset_view, name="generate"),

    path("", dashboard_view, name="dashboard"),

    # Analytic
    path("analysis/", analysis_list_view, name="analysis_list"),

    path("analysis/create/", create_analysis_view, name="create_analysis"),

    path("analysis/<int:session_id>/",
         analysis_dashboard, name="analysis_dashboard"),

    # path("analysis/<int:session_id>/step/signal/",
    #      step_signal_view, name="step_signal"),

    path("analysis/<int:session_id>/api/compute_tau/", api_compute_tau),
    path("analysis/<int:session_id>/api/compute_dimension/", api_compute_dimension),
    path("analysis/<int:session_id>/api/compute_lyapunov/", api_compute_lyapunov),
    path("analysis/<int:session_id>/api/reconstruct/", api_reconstruct),

    # ML
    path("ml/", ml_dashboard, name="ml_dashboard"),
    path("meta/", ml_meta_view, name="ml_meta"),
    path("train/", train_model_view, name="train_model"),

    # Predict
    path("predict/", predict_dashboard, name="predict_dashboard"),
    path("predict/run/", predict_run),

]
