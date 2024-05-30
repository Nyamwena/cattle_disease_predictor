# Generated by Django 5.0.6 on 2024-05-28 20:47

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("predictor", "0003_prediction"),
    ]

    operations = [
        migrations.AlterField(
            model_name="prediction",
            name="disease",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="predictor.disease",
            ),
        ),
    ]