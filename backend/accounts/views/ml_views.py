import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from django.utils.safestring import mark_safe
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.db.models import Q

from ..models import Dataset, TrainedModel, AnalysisSession

from ..ml.models_registry import MODELS, MODEL_REGISTRY
from ..ml.models import *
# from accounts.ml.train_service import train_lstm


@login_required
def ml_dashboard(request):

    datasets = Dataset.objects.filter(user=request.user)

    datasets_data = []

    for ds in datasets:
        df = pd.read_csv(ds.file.path)

        datasets_data.append({
            "id": ds.id,
            "title": ds.title,
            "columns": list(df.columns)
        })

    return render(request, "ml/dashboard.html", {
        "datasets": datasets,
        "datasets_json": json.dumps(datasets_data),
        "models_json": json.dumps(MODELS)
    })


@login_required
def ml_meta_view(request):

    dataset_id = request.GET.get("dataset_id")

    df = pd.read_csv(
        Dataset.objects.get(id=dataset_id).file.path
    )

    return JsonResponse({
        "columns": list(df.columns)
    })


@login_required
def train_model_view(request):

    # -------------------------
    # 1. INPUTS
    # -------------------------
    dataset_id = request.POST.get("dataset_id")
    target = request.POST.get("target")
    model_name = request.POST.get("model_name", "lstm").lower()

    seq_len = int(request.POST.get("sequence_length", 20))
    hidden_size = int(request.POST.get("hidden_size", 64))
    num_layers = int(request.POST.get("num_layers", 2))
    lr = float(request.POST.get("learning_rate", 0.001))
    epochs = int(request.POST.get("epochs", 10))

    # -------------------------
    # 2. LOAD DATA
    # -------------------------
    dataset = Dataset.objects.get(id=dataset_id)
    df = pd.read_csv(dataset.file.path)

    if target not in df.columns:
        return JsonResponse({"error": "Invalid target column"}, status=400)

    data = df.values.astype(np.float32)
    target_idx = df.columns.get_loc(target)

    # normalize
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    data = (data - mean) / std

    # -------------------------
    # 3. SEQUENCES
    # -------------------------
    X, y = create_sequences(data, target_idx, seq_len)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    split = int(len(X) * 0.8)

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # -------------------------
    # 4. MODEL FACTORY
    # -------------------------
    model_class = MODEL_REGISTRY.get(model_name)

    if model_class is None:
        return JsonResponse({"error": "Unknown model"}, status=400)

    model = model_class(
        input_size=X.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # 5. TRAIN LOOP
    # -------------------------
    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        pred = model(X_train)
        loss = criterion(pred, y_train)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # validation
        model.eval()
        with torch.no_grad():

            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        val_losses.append(val_loss.item())

    # -------------------------
    # 6. METRICS
    # -------------------------
    with torch.no_grad():
        pred = model(X_val)

    mse = float(criterion(pred, y_val).item())
    rmse = float(np.sqrt(mse))

    nrmse = rmse / (y_val.max().item() - y_val.min().item() + 1e-8)

    emse = mse * 1.1  # simplified heuristic

    accuracy = float(1 - (rmse / (np.std(y_val.numpy()) + 1e-8)))

    # -------------------------
    # 7. SAVE MODEL
    # -------------------------
    trained_model = TrainedModel.objects.create(
        user=request.user,
        dataset=dataset,
        name=f"{dataset.title}_{model_name}"
    )

    acc_int = int(accuracy * 100)

    analysis = AnalysisSession.objects.filter(
        Q(reconstructed_dataset=dataset.id) |
        Q(dataset=dataset),
        user=request.user,
    ).first()

    lyap = None

    if analysis:
        lyap = analysis.lyapunov_exponent

    forecast_horizon = None

    if lyap is not None and lyap > 0:
        forecast_horizon = round(np.log(1 / nrmse) / lyap)

    model_path = f"trained_models/weights/model_{trained_model.id}_acc{acc_int}.pt"
    loss_path = f"trained_models/rates/loss_{trained_model.id}_acc{acc_int}.json"
    params_path = f"trained_models/parametres/params_{trained_model.id}_acc{acc_int}.json"

    full_model_path = os.path.join(settings.MEDIA_ROOT, model_path)
    full_loss_path = os.path.join(settings.MEDIA_ROOT, loss_path)
    full_params_path = os.path.join(settings.MEDIA_ROOT, params_path)

    # ensure dirs
    os.makedirs(os.path.dirname(full_model_path), exist_ok=True)

    # save weights
    torch.save(model.state_dict(), full_model_path)

    # save losses
    with open(full_loss_path, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f)

    # save params
    with open(full_params_path, "w") as f:
        json.dump({
            "model": model_name,
            "sequence_length": seq_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "learning_rate": lr,
            "epochs": epochs,
            "target": target,

            "mse": mse,
            "rmse": rmse,
            "nrmse": float(nrmse),
            "emse": float(emse),
            "accuracy": float(accuracy),

            "forecast_horizon": forecast_horizon,

            "train_losses": train_losses,
            "val_losses": val_losses,

            "model_id": trained_model.id
        }, f)

    # attach to DB (IMPORTANT)
    trained_model.weights.name = model_path
    trained_model.rates.name = loss_path
    trained_model.model_parametres.name = params_path
    trained_model.save()

    trained_model.name = f'{trained_model.name}_{trained_model.created_at.strftime("%d.%m.%Y %H:%M")}'
    trained_model.save()

    # -------------------------
    # 8. RESPONSE
    # -------------------------
    return JsonResponse({

        "mse": mse,
        "rmse": rmse,
        "nrmse": float(nrmse),
        "emse": float(emse),
        "accuracy": float(accuracy),

        "train_losses": train_losses,
        "val_losses": val_losses,

        "model_id": trained_model.id
    })


# @login_required
# def save_model_view(request):

#     model_id = request.POST.get("model_id")

#     # TODO: save weights + params

#     return JsonResponse({
#         "status": "ok"
#     })


# @login_required
# def train_model_view(request):

#     dataset_id = request.POST.get("dataset")

#     target = request.POST.get("target")

#     dataset = Dataset.objects.get(id=dataset_id)

#     df = pd.read_csv(dataset.file.path)

#     model = request.POST.get("model_name")

#     params = {

#         "sequence_length":
#             int(request.POST.get("sequence_length")),

#         "hidden_size":
#             int(request.POST.get("hidden_size")),

#         "num_layers":
#             int(request.POST.get("num_layers")),

#         "learning_rate":
#             float(request.POST.get("learning_rate")),

#         "epochs":
#             int(request.POST.get("epochs")),

#         "batch_size":
#             int(request.POST.get("batch_size"))
#     }

#     if model == "LSTM":
#         result = train_lstm(
#             df,
#             target,
#             params
#         )

#     return JsonResponse(result)
