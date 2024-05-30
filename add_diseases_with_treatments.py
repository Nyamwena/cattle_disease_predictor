import os
import django
import pandas as pd

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cattle_disease_predictor.settings')
django.setup()

from predictor.models import Disease

# Path to the CSV file
file_path = 'cattle_diseases.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Define common diseases and their treatments
disease_treatments = {
    'mastitis': 'Antibiotic therapy, anti-inflammatory drugs, frequent milking to remove infected milk.',
    'foot and mouth disease': 'There is no specific treatment; supportive care includes rest, isolation, and soft diet. Vaccination is preventive.',
    'bovine respiratory disease': 'Antibiotics, anti-inflammatory drugs, supportive care (fluids, nutritional support).',
    'bovine tuberculosis': 'No treatment; infected animals are usually culled to prevent spread.',
    'bovine viral diarrhea': 'Supportive care, antibiotics for secondary infections, vaccination is preventive.',
    'lumpy skin disease': 'Supportive care, antibiotics for secondary infections, vaccination is preventive.',
    'anaplasmosis': 'Antibiotics (tetracyclines), blood transfusions in severe cases, supportive care.',
    'anthrax': 'High-dose antibiotics (penicillin, tetracyclines), vaccination is preventive.',
    'blackleg': 'Antibiotics (penicillin), vaccination is preventive.',
    'brucellosis': 'No treatment; infected animals are usually culled. Vaccination is preventive.',
    'calf scours': 'Fluid and electrolyte replacement, antibiotics if bacterial infection is present, supportive care.',
    'coccidiosis': 'Anticoccidial drugs (sulfa drugs, amprolium), supportive care, proper sanitation.',
    'dermatophytosis (ringworm)': 'Antifungal treatments (topical iodine, thiabendazole), good hygiene practices.',
    'frostbite': 'Gradual warming, antibiotics for secondary infections, supportive care.',
    'hardware disease': 'Magnets to prevent hardware from causing damage, surgical removal of foreign objects, antibiotics for secondary infections.',
    'heat stress': 'Provide shade, water, and ventilation, reduce handling, electrolyte supplements.',
    'johne’s disease': 'No effective treatment; infected animals are usually culled to prevent spread.',
    'ketosis': 'Intravenous glucose, propylene glycol drench, corticosteroids, supportive care.',
    'listeriosis': 'Antibiotics (penicillin, tetracyclines), supportive care.',
    'milk fever (hypocalcemia)': 'Intravenous calcium gluconate, oral calcium supplements, supportive care.',
    'pinkeye': 'Antibiotics (oxytetracycline), eye patches or protective masks, fly control measures.',
    'pneumonia': 'Antibiotics, anti-inflammatory drugs, supportive care (fluids, nutritional support).',
    'rabies': 'No treatment; infected animals are euthanized to prevent spread. Vaccination is preventive.',
    'salmonellosis': 'Antibiotics, fluid therapy, supportive care, good sanitation practices.',
    'shipping fever': 'Antibiotics, anti-inflammatory drugs, supportive care (fluids, nutritional support).',
    'tetanus': 'Antibiotics, antitoxin, wound care, supportive care. Vaccination is preventive.',
    'trichomoniasis': 'No treatment; infected animals are usually culled to prevent spread. Proper herd management and testing.',
    'vibriosis': 'Antibiotics, vaccination is preventive, proper herd management.',
    'white muscle disease': 'Supplementation with selenium and vitamin E, supportive care.'
    # Add other diseases and treatments as needed
}

# Extract unique disease names from the 'prognosis' column
diseases = data['prognosis'].unique()

# Add diseases to the database with their treatments
for disease_name in diseases:
    treatment = disease_treatments.get(disease_name.lower(), 'Treatment information not available.')
    disease, created = Disease.objects.get_or_create(name=disease_name, defaults={'treatment': treatment})
    if created:
        print(f'Added disease: {disease_name}')
    else:
        print(f'Disease already exists: {disease_name}')
