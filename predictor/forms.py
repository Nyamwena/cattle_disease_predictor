from django import forms
from .models import Symptom, TheileriosisSymptom

class SymptomForm(forms.Form):
    cow_id = forms.CharField(max_length=50)
    cow_name = forms.CharField(max_length=100)
    location = forms.CharField(max_length=100)
    season = forms.CharField(max_length=50)
    symptoms = forms.ModelMultipleChoiceField(queryset=Symptom.objects.all(), widget=forms.CheckboxSelectMultiple)

class TheileriosisSymptomForm(forms.Form):
    cow_id = forms.CharField(max_length=50)
    cow_name = forms.CharField(max_length=100)
    location = forms.CharField(max_length=100)
    season = forms.CharField(max_length=50)
    symptoms = forms.ModelMultipleChoiceField(queryset=TheileriosisSymptom.objects.all(), widget=forms.CheckboxSelectMultiple)
