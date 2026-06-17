from django.db import models
from django.contrib.auth.models import User


class Dataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="datasets/")
    created_at = models.DateTimeField(auto_now_add=True)


class AnalysisSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    signal_parameter = models.CharField(
        max_length=255, null=True, blank=True
    )
    tau = models.IntegerField(null=True, blank=True)
    embedding_dim = models.IntegerField(null=True, blank=True)
    lyapunov_exponent = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=255)
    reconstructed_dataset = models.ForeignKey(
        Dataset, on_delete=models.SET_NULL,
        related_name="reconstructed_analysis_sessions",
        null=True, blank=True
    )


class TrainedModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    model_parametres = models.FileField(
        upload_to="trained_models/parametres"
    )
    weights = models.FileField(
        upload_to="trained_models/weights"
    )
    rates = models.FileField(
        upload_to="trained_models/rates"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=255)


class PredictionModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE)
    prediction = models.FileField(
        upload_to="trained_models/prediction"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=255)
    description = models.CharField(
        max_length=255, null=True, blank=True
    )
