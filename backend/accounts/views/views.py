import pandas as pd
import plotly.express as px
import os
import tempfile
import numpy as np

from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.shortcuts import get_object_or_404
from django.contrib import messages
from django.core.files import File
from datetime import datetime
from ..forms import RegisterForm, LoginForm, DatasetUploadForm, Dataset
from ..generators.registry import ATTRACTORS


def register_view(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)

        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = RegisterForm()

    return render(request, "accounts/register.html", {
        "form": form
    })


class CustomLoginView(LoginView):
    authentication_form = LoginForm
    template_name = "accounts/login.html"


@login_required
def dashboard_view(request):
    return render(request, "accounts/dashboard.html")


@login_required
def upload_dataset(request):

    preview = None

    if request.method == "POST":
        form = DatasetUploadForm(request.POST, request.FILES)

        if form.is_valid():

            dataset = form.save(commit=False)
            dataset.user = request.user
            dataset.save()

            df = pd.read_csv(dataset.file.path)

            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = pd.factorize(df[col])[0]

            df.to_csv(
                dataset.file.path,
                index=False
            )

            preview = df.head().to_html(
                classes="table table-striped",
                index=False
            )
            return redirect(
                "dataset_detail",
                pk=dataset.id
            )

    else:
        form = DatasetUploadForm()

    return render(request, "accounts/upload.html", {
        "form": form
    })


@login_required
def delete_dataset(request, pk):

    dataset = get_object_or_404(
        Dataset,
        id=pk,
        user=request.user
    )

    if dataset.file:
        if os.path.exists(dataset.file.path):
            os.remove(dataset.file.path)

    dataset.delete()

    return redirect("datasets")


@login_required
def datasets_view(request):

    datasets = Dataset.objects.filter(
        user=request.user
    ).order_by("-created_at")

    return render(
        request,
        "accounts/datasets.html",
        {
            "datasets": datasets
        }
    )


@login_required
def dataset_detail_view(request, pk):

    dataset = get_object_or_404(
        Dataset,
        pk=pk,
        user=request.user
    )

    df = pd.read_csv(dataset.file.path)

    preview = df.head().to_html(
        classes="table table-striped",
        index=False
    )

    stats = df.describe().to_html(
        classes="table table-bordered"
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    target = request.GET.get("target")

    plot_div = None

    if target in numeric_columns:

        fig = px.line(
            df,
            y=target,
            title=f"{target} Временной ряд"
        )

        plot_div = fig.to_html(
            full_html=False
        )

    return render(
        request,
        "accounts/dataset_detail.html",
        {
            "dataset": dataset,
            "preview": preview,
            "stats": stats,
            "columns": numeric_columns,
            "target": target,
            "plot_div": plot_div,
        }
    )


# @login_required
# def dataset_detail_view(request, pk):
#     dataset = Dataset.objects.get(pk=pk)

#     df = pd.read_csv(dataset.file.path)

#     columns = df.columns.tolist()

#     return render(request, "datasets/detail.html", {
#         "dataset": dataset,
#         "columns": columns
#     })

@login_required
def generate_dataset_view(request):
    generator_map = ATTRACTORS

    schemas = {
        name: gen.params_schema()
        for name, gen in generator_map.items()
    }

    if request.method == "POST":
        name = request.POST.get("attractor")
        generator = generator_map.get(name)

        if not generator:
            messages.error(request, "Неизвестный генератор")
            return render(request, "datasets/generate.html", {
                "generators": generator_map,
                "schemas": schemas,
            })

        dataset_title = request.POST.get("dataset_name", "").strip()

        # Получаем параметры
        schema = generator.params_schema()

        def cast(value, type_name):
            if type_name == "int":
                return int(float(value))
            if type_name == "float":
                return float(value)
            return value

        params = {}
        for key, meta in schema.items():
            value = request.POST.get(key, meta["default"])
            params[key] = cast(value, meta["type"])

        # Генерируем данные
        data = generator.generate(**params)

        # Создаем DataFrame
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame({'value': data})
            elif data.ndim == 2:
                # Для 3D аттракторов (x, y, z)
                if data.shape[1] == 3:
                    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
                elif data.shape[1] == 2:
                    df = pd.DataFrame(data, columns=['x', 'y'])
                else:
                    df = pd.DataFrame(
                        data, columns=[f'col_{i}' for i in range(data.shape[1])])
            else:
                messages.error(request, "Неподдерживаемая размерность данных")
                return render(request, "datasets/generate.html", {
                    "generators": generator_map,
                    "schemas": schemas,
                })
        else:
            df = pd.DataFrame(data)

        dataset = Dataset.objects.create(
            user=request.user,
            title=dataset_title
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{dataset_title}_{timestamp}.csv"

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file.flush()

            with open(tmp_file.name, 'rb') as f:
                dataset.file.save(filename, File(f), save=True)

            os.unlink(tmp_file.name)

        # Перенаправляем на страницу деталей датасета
        return redirect("dataset_detail", pk=dataset.id)

    return render(request, "datasets/generate.html", {
        "generators": generator_map,
        "schemas": schemas,
    })

# def generate_dataset_view(request):
#     generator_map = ATTRACTORS

#     schemas = {
#         name: gen.params_schema()
#         for name, gen in generator_map.items()
#     }

#     data = None

#     if request.method == "POST":
#         name = request.POST.get("attractor")
#         generator = generator_map[name]

#         schema = generator.params_schema()

#         def cast(value, type_name):
#             if type_name == "int":
#                 return int(float(value))
#             if type_name == "float":
#                 return float(value)
#             return value

#         params = {}

#         for key, meta in schema.items():
#             value = request.POST.get(key, meta["default"])
#             params[key] = cast(value, meta["type"])

#         data = generator.generate(**params)


#     return render(request, "datasets/generate.html", {
#         "generators": generator_map,
#         "schemas": schemas,
#         "data": data
#     })
