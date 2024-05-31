from django.db import models

class Symptom(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Disease(models.Model):
    name = models.CharField(max_length=100)
    treatment = models.TextField()

    def __str__(self):
        return self.name

class TheileriosisSymptom(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Prediction(models.Model):
    cow_id = models.CharField(max_length=50)
    cow_name = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    season = models.CharField(max_length=50)
    date = models.DateTimeField(auto_now_add=True)
    disease = models.ForeignKey(Disease, null=True, blank=True, on_delete=models.CASCADE)
    has_theileriosis = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.cow_name} - {self.disease.name if self.disease else "Theileriosis"}'


class PredictionStage(models.Model):
    cow_name = models.CharField(max_length=100)
    cow_id = models.CharField(max_length=100)
    predicted_stage = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.cow_name} ({self.cow_id}) - {self.predicted_stage}'