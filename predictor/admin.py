from django.contrib import admin
from .models import Symptom, Disease, TheileriosisSymptom, Prediction

# Register your models here
admin.site.register(Symptom)
admin.site.register(Disease)
admin.site.register(TheileriosisSymptom)
admin.site.register(Prediction)

