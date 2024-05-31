# Generated by Django 5.0.6 on 2024-05-31 09:02

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("predictor", "0004_alter_prediction_disease"),
    ]

    operations = [
        migrations.CreateModel(
            name="PredictionStage",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("cow_name", models.CharField(max_length=100)),
                ("cow_id", models.CharField(max_length=100)),
                ("predicted_stage", models.CharField(max_length=100)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
