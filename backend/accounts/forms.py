from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from .models import Dataset

class LoginForm(AuthenticationForm):
        username = forms.CharField(label="Логин")
        password = forms.CharField(label="Пароль", widget=forms.PasswordInput)

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ["title", "file"]