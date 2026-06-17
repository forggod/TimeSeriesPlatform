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

from ..models import Dataset, TrainedModel, PredictionModel

from ..ml.models_registry import MODELS, MODEL_REGISTRY
from ..ml.models import *


@login_required
def predict_dashboard(request):

    models_data = []

    for model in TrainedModel.objects.filter(
        user=request.user
    ). order_by("-created_at"):

        if not model.model_parametres:
            continue

        with open(model.model_parametres.path) as f:
            params = json.load(f)

            models_data.append({
                "id": model.id,
                "name": model.name,
                "created_at": model.created_at.strftime("%d.%m.%Y %H:%M"),
                "accuracy": params.get("accuracy"),
                "forecast_horizon": params.get("forecast_horizon"),
                "target": params.get("target"),
                "model_type": params.get("model")
            })

    return render(request, "predict/dashboard.html", {
        "models_json": json.dumps(models_data),
    })


@login_required
def predict_run(request):

    model_id = request.POST.get("model_id")
    max_steps = int(request.POST.get("max_steps"))

    trained_model = TrainedModel.objects.get(id=model_id)
    params = json.load(open(trained_model.model_parametres.path))

    seq_len = params["sequence_length"]
    target = params["target"]
    model_name = params["model"]

    df = pd.read_csv(trained_model.dataset.file.path)
    data = df.values.astype(np.float32)

    target_idx = df.columns.get_loc(target)

    # нормализация (упрощённо)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    data = (data - mean) / std

    model_class = MODEL_REGISTRY[model_name]

    model = model_class(
        input_size=data.shape[1],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"]
    )

    model.load_state_dict(torch.load(
        trained_model.weights.path, map_location="cpu"))
    model.eval()

    # стартовое окно
    window = data[-seq_len*4:].copy()

    history = list(window[:, target_idx])  # 👈 ВАЖНО: что рисуем как историю
    predictions = []

    for step in range(max_steps):

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(x).item()

        predictions.append(float(pred))

        # autoregressive update
        new_row = window[-1].copy()
        new_row[target_idx] = pred

        window = np.vstack([window[1:], new_row])

    prediction_data = {
        "history": [float(x) for x in history],
        "predicted": [float(x) for x in predictions]
    }

    # создаём объект
    prediction_obj = PredictionModel.objects.create(
        user=request.user,
        model=trained_model,
        name=f"prediction_{trained_model.id}",
        description="Auto-generated prediction"
    )

    # путь к файлу
    file_name = f"prediction_{prediction_obj.id}.json"
    file_path = os.path.join(
        settings.MEDIA_ROOT,
        "trained_models",
        "prediction",
        file_name
    )

    # создаём директорию если нет
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # сохраняем JSON
    with open(file_path, "w") as f:
        json.dump(prediction_data, f)

    # привязываем файл к модели Django
    prediction_obj.prediction.name = f"trained_models/prediction/{file_name}"
    prediction_obj.save()

    return JsonResponse({
        "history": [float(x) for x in history],
        "predicted": predictions
    })
