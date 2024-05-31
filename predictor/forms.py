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


class PredictionForm(forms.Form):
    cow_name = forms.CharField(max_length=100)
    cow_id = forms.CharField(max_length=100)

    FEVER_CHOICES = [(round(x, 1), round(x, 1)) for x in range(37, 42)]
    RESPIRATORY_SIGNS_CHOICES = [
        ('Mild', 'Mild'),
        ('Moderate', 'Moderate'),
        ('Severe', 'Severe')
    ]
    YES_NO_CHOICES = [('Yes', 'Yes'), ('No', 'No')]

    fever = forms.ChoiceField(choices=FEVER_CHOICES)
    lymph_node_swelling = forms.ChoiceField(choices=YES_NO_CHOICES)
    appetite_loss = forms.ChoiceField(choices=YES_NO_CHOICES)
    lethargy = forms.ChoiceField(choices=YES_NO_CHOICES)
    respiratory_signs = forms.ChoiceField(choices=RESPIRATORY_SIGNS_CHOICES)
    anemia = forms.ChoiceField(choices=YES_NO_CHOICES)
    jaundice = forms.ChoiceField(choices=YES_NO_CHOICES)
    weight_loss = forms.ChoiceField(choices=YES_NO_CHOICES)
    reproductive_issues = forms.ChoiceField(choices=YES_NO_CHOICES)