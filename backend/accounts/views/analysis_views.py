import json
import pandas as pd

from django.contrib.auth.decorators import login_required
from django.shortcuts import (
    render, redirect, get_object_or_404
)
from django.http import JsonResponse
from django.core.files.base import ContentFile

from ..models import Dataset, AnalysisSession
from ..services.analytic_service import (
    compute_acf, compute_mif,
    compute_dim_auto_correlation,
    compute_dim_gamma_test,
    compute_dim_false_nearest_neighbors,
    compute_lyapunov,
    reconstruct_attractor
)


@login_required
def analysis_list_view(request):
    datasets = Dataset.objects.filter(user=request.user)
    sessions = AnalysisSession.objects.filter(user=request.user)

    sessions_data = [
        {
            "id": s.id,
            "dataset_id": s.dataset_id,
            "name": s.name,
            "created_at": s.created_at.strftime("%Y-%m-%d %H:%M"),
            "signal_parameter": s.signal_parameter,
            "tau": s.tau,
            "embedding_dim": s.embedding_dim,
            "lyapunov_exponent": s.lyapunov_exponent,
        }
        for s in sessions
    ]

    return render(request, "analytics/list.html", {
        "datasets": datasets,
        "sessions_json": json.dumps(sessions_data)
    })


@login_required
def create_analysis_view(request):
    if request.method == "POST":
        dataset_id = request.POST.get("dataset_id")

        session = AnalysisSession.objects.create(
            user=request.user,
            dataset_id=dataset_id
        )

        session.save()

        return JsonResponse({
            "session_id": session.id
        })


@login_required
def analysis_dashboard(request, session_id):
    session = get_object_or_404(
        AnalysisSession, id=session_id, user=request.user)

    df = pd.read_csv(session.dataset.file.path)
    columns = df.columns.to_list()

    return render(request, "analytics/dashboard.html", {
        "session": session,
        "columns": columns,
        "signal": session.signal_parameter,
        "tau": session.tau,
        "dimension": session.embedding_dim,
        "lyapunov": session.lyapunov_exponent
    })


@login_required
def api_compute_tau(request, session_id):
    session = get_object_or_404(
        AnalysisSession,
        id=session_id,
        user=request.user
    )

    signal = request.POST.get("signal")

    df = pd.read_csv(session.dataset.file.path)
    series = df[signal].values

    acf, auto_tau = compute_acf(series)
    mif, _ = compute_mif(series)

    return JsonResponse({
        "acf": {
            "x": list(range(len(acf))),
            "y": acf.tolist()
        },
        "mif": {
            "x": list(range(len(mif))),
            "y": mif.tolist()
        },
        "tau": auto_tau
    })


@login_required
def api_compute_dimension(request, session_id):
    session = get_object_or_404(
        AnalysisSession,
        id=session_id,
        user=request.user
    )

    signal = request.POST.get("signal")
    tau = int(request.POST.get("tau"))

    df = pd.read_csv(session.dataset.file.path)
    series = df[signal].values

    auto_dim_corr, corr_values = compute_dim_auto_correlation(
        series, tau
    )

    auto_gamma, gamma_values = compute_dim_gamma_test(
        series, tau
    )

    auto_fnn, fnn_values = compute_dim_false_nearest_neighbors(
        series,
        tau
    )

    return JsonResponse({
        "auto_corr": {
            "x": list(range(len(corr_values)),),
            "y": corr_values.tolist(),
            "auto_value": int(auto_dim_corr)
        },
        "gamma_test": {
            "x": list(range(len(gamma_values)),),
            "y": gamma_values.tolist(),
            "auto_value": int(auto_gamma)
        },
        "false_nearest": {
            "x": list(range(len(fnn_values)),),
            "y": fnn_values,
            "auto_value": int(auto_fnn)
        }
    })


@login_required
def api_compute_lyapunov(request, session_id):
    session = get_object_or_404(
        AnalysisSession,
        id=session_id,
        user=request.user
    )

    signal = request.POST.get("signal")
    tau = int(request.POST.get("tau"))
    dim = int(request.POST.get("dimension"))

    df = pd.read_csv(session.dataset.file.path)
    series = df[signal].values

    value, values = compute_lyapunov(
        series, tau, dim
    )

    return JsonResponse({
        "lyapunov": {
            "x": list(range(len(values)),),
            "y": values,
            "value": round(float(value), 6)
        }
    })


@login_required
def api_reconstruct(request, session_id):
    session = get_object_or_404(
        AnalysisSession,
        id=session_id,
        user=request.user
    )

    signal = request.POST.get("signal")
    tau = int(request.POST.get("tau"))
    dimension = int(request.POST.get("dimension"))
    lyapunov = float(request.POST.get("lyapunov"))

    session.signal_parameter = signal
    session.tau = tau
    session.embedding_dim = dimension
    session.lyapunov_exponent = lyapunov
    session.save()

    df = pd.read_csv(session.dataset.file.path)
    series = df[signal].values

    rec_attractor = reconstruct_attractor(series, tau=tau, dim=dimension)

    cols = [f"x{i}" for i in range(rec_attractor.shape[1])]
    new_df = pd.DataFrame(rec_attractor, columns=cols)

    csv_bytes = new_df.to_csv(index=False).encode("utf-8")

    dataset = Dataset.objects.create(
        user=session.user,
        title=f"{session.dataset.title} reconstructed_{signal}",
    )

    dataset.title += f" {dataset.id}"

    dataset.file.save(
        f"reconstructed_{dataset.id}.csv",
        ContentFile(csv_bytes)
    )

    dataset.save()

    session.reconstructed_dataset = dataset
    session.save()

    return JsonResponse({
        "status": "ok",
        "name": dataset.title,
        "dataset_id": dataset.id
    })
