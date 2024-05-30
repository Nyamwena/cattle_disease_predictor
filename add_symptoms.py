import os
import django
import pandas as pd

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cattle_disease_predictor.settings')
django.setup()

from predictor.models import Symptom

# Path to the CSV file
file_path = 'cattle_diseases.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Extract symptom names (excluding the prognosis column)
symptoms = list(data.columns[:-1])

# Add symptoms to the database
for symptom_name in symptoms:
    symptom, created = Symptom.objects.get_or_create(name=symptom_name)
    if created:
        print(f'Added symptom: {symptom_name}')
    else:
        print(f'Symptom already exists: {symptom_name}')
