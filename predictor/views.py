from django.shortcuts import render
import joblib
import numpy as np
from .forms import SymptomForm, TheileriosisSymptomForm, PredictionForm
from .models import Disease, Symptom, TheileriosisSymptom, Prediction, PredictionStage
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import io
import urllib, base64
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
import os
from django.http import JsonResponse

# Load the trained models
general_model_path = 'databest_cattle_disease_model.pkl'
theileriosis_model_path = 'theileriosis_model.pkl'

model_path = 'theileriosis_stage_model.joblib'
encoders_path = 'label_encoders.joblib'
stage_encoder_path = 'stage_encoder.joblib'

general_model = joblib.load(general_model_path)
theileriosis_model = joblib.load(theileriosis_model_path)

@login_required
def predict_general_disease(symptoms):
    input_vector = np.zeros(len(Symptom.objects.all()))
    for symptom in symptoms:
        input_vector[symptom.id - 1] = 1
    prediction = general_model.predict([input_vector])[0]
    return prediction

@login_required
def predict_theileriosis(symptoms):
    input_vector = np.zeros(len(TheileriosisSymptom.objects.all()))
    for symptom in symptoms:
        input_vector[symptom.id - 1] = 1
    prediction = theileriosis_model.predict([input_vector])[0]
    return prediction

@login_required
def index(request):
    return render(request, 'predictor/index.html')


def predict_general(request):
    if request.method == 'POST':
        print("predict_general view called")
        form = SymptomForm(request.POST)
        if form.is_valid():
            print("Form is valid")
            cow_id = form.cleaned_data['cow_id']
            cow_name = form.cleaned_data['cow_name']
            location = form.cleaned_data['location']
            season = form.cleaned_data['season']
            symptoms = form.cleaned_data['symptoms']
            # symptom_ids = [symptom.id for symptom in symptoms]  # Extract symptom IDs

            # print(
                # f"Form cleaned data: cow_id={cow_id}, cow_name={cow_name}, location={location}, season={season}, symptoms={symptom_ids}")

            all_symptoms = list(Symptom.objects.all())
            input_vector = np.zeros(len(all_symptoms))
            for symptom in symptoms:
                index = all_symptoms.index(symptom)
                input_vector[index] = 1

            prediction = general_model.predict([input_vector])[0]
            disease = Disease.objects.get(name=prediction)

            # Save prediction to the database
            Prediction.objects.create(
                cow_id=cow_id,
                cow_name=cow_name,
                location=location,
                season=season,
                disease=disease
            )

            context = {
                'form': form,
                'disease': disease.name,
                'treatment': disease.treatment
            }
            return render(request, 'predictor/result.html', context)
    else:
        form = SymptomForm()
    return render(request, 'predictor/predict_general.html', {'form': form})


@login_required
def predict_theileriosis(request):
    if request.method == 'POST':
        form = TheileriosisSymptomForm(request.POST)
        if form.is_valid():
            cow_id = form.cleaned_data['cow_id']
            cow_name = form.cleaned_data['cow_name']
            location = form.cleaned_data['location']
            season = form.cleaned_data['season']
            symptoms = form.cleaned_data['symptoms']

            # Create an input vector for the model based on the selected symptoms
            all_symptoms = list(TheileriosisSymptom.objects.all())
            input_vector = np.zeros(len(all_symptoms))
            for symptom in symptoms:
                index = all_symptoms.index(symptom)
                input_vector[index] = 1

            # Make a prediction
            print("Input vector for Theileriosis prediction:", input_vector)  # Debugging statement
            prediction = theileriosis_model.predict([input_vector])[0]
            print("Theileriosis model prediction:", prediction)  # Debugging statement

            # Determine if the prediction indicates Theileriosis
            has_theileriosis = prediction == "Positive"

            # Save prediction to the database
            Prediction.objects.create(
                cow_id=cow_id,
                cow_name=cow_name,
                location=location,
                season=season,
                has_theileriosis=has_theileriosis
            )

            context = {
                'form': form,
                'has_theileriosis': has_theileriosis,
                'treatment': 'Buparvaquone (primary drug), oxytetracycline, antipyretics, haematinics, antihistaminics, ruminotorics, liver tonics' if has_theileriosis else 'No treatment necessary'
            }
            return render(request, 'predictor/theileriosis_result.html', context)
    else:
        form = TheileriosisSymptomForm()
    return render(request, 'predictor/predict_theileriosis.html', {'form': form})

@login_required
def prediction_history(request):
    predictions = Prediction.objects.all().order_by('-date')
    return render(request, 'predictor/prediction_history.html', {'predictions': predictions})


@login_required
def plot_predictions_by_location(request):
    predictions = Prediction.objects.all()
    data = {}
    for prediction in predictions:
        if prediction.location not in data:
            data[prediction.location] = 0
        data[prediction.location] += 1

    locations = list(data.keys())
    counts = list(data.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=locations, y=counts)
    plt.title('Predictions by Location')
    plt.xlabel('Location')
    plt.ylabel('Number of Predictions')
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    return render(request, 'predictor/plot.html', {'data': uri})


@login_required
def plot_predictions_by_season(request):
    predictions = Prediction.objects.all()
    data = {}
    for prediction in predictions:
        if prediction.season not in data:
            data[prediction.season] = 0
        data[prediction.season] += 1

    seasons = list(data.keys())
    counts = list(data.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=seasons, y=counts)
    plt.title('Predictions by Season')
    plt.xlabel('Season')
    plt.ylabel('Number of Predictions')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    return render(request, 'predictor/plot.html', {'data': uri})


@login_required
def plot_heatmap(request):
    predictions = Prediction.objects.all()
    data = {
        'Location': [],
        'Season': [],
        'Count': []
    }
    for prediction in predictions:
        data['Location'].append(prediction.location)
        data['Season'].append(prediction.season)
        data['Count'].append(1)

    df = pd.DataFrame(data)
    pivot_table = df.pivot_table(index='Location', columns='Season', aggfunc='size', fill_value=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
    plt.title('Heatmap of Predictions by Location and Season')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    return render(request, 'predictor/plot.html', {'data': uri})


@login_required
def plot_predictions_over_time(request):
    predictions = Prediction.objects.all()
    data = {
        'Date': [],
        'Count': []
    }
    for prediction in predictions:
        data['Date'].append(prediction.date.date())
        data['Count'].append(1)

    df = pd.DataFrame(data)
    df = df.groupby('Date').sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Count', data=df)
    plt.title('Predictions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Predictions')
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    return render(request, 'predictor/plot.html', {'data': uri})


@login_required
def plot_predictions_by_disease(request):
    predictions = Prediction.objects.all()
    data = {
        'Disease': [],
        'Count': []
    }
    for prediction in predictions:
        if prediction.disease:
            data['Disease'].append(prediction.disease.name)
        else:
            data['Disease'].append('Theileriosis')
        data['Count'].append(1)

    df = pd.DataFrame(data)
    df = df.groupby('Disease').sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.pie(df['Count'], labels=df['Disease'], autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Predictions by Disease')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    return render(request, 'predictor/plot.html', {'data': uri})


@login_required
def plot_scatter_location_season(request):
    predictions = Prediction.objects.all()
    data = {
        'Location': [],
        'Season': [],
        'Count': []
    }
    for prediction in predictions:
        data['Location'].append(prediction.location)
        data['Season'].append(prediction.season)
        data['Count'].append(1)

    df = pd.DataFrame(data)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Location', y='Season', size='Count', data=df, sizes=(20, 200), legend=False)
    plt.title('Predictions by Location and Season')
    plt.xlabel('Location')
    plt.ylabel('Season')
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    return render(request, 'predictor/plot.html', {'data': uri})


model = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)
stage_encoder = joblib.load(stage_encoder_path)

def predict_stage(request):
    prediction = None
    predicted_stage = None  # Initialize predicted_stage
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            cow_name = form.cleaned_data['cow_name']
            cow_id = form.cleaned_data['cow_id']

            # Extract features from form
            features = {
                'Fever': float(form.cleaned_data['fever']),
                'Lymph_Node_Swelling': form.cleaned_data['lymph_node_swelling'],
                'Appetite_Loss': form.cleaned_data['appetite_loss'],
                'Lethargy': form.cleaned_data['lethargy'],
                'Respiratory_Signs': form.cleaned_data['respiratory_signs'],
                'Anemia': form.cleaned_data['anemia'],
                'Jaundice': form.cleaned_data['jaundice'],
                'Weight_Loss': form.cleaned_data['weight_loss'],
                'Reproductive_Issues': form.cleaned_data['reproductive_issues']
            }

            # Encode categorical features
            for column, le in label_encoders.items():
                features[column] = le.transform([features[column]])[0]

            # Prepare feature array
            feature_array = np.array([list(features.values())])

            # Predict the stage
            prediction = model.predict(feature_array)
            predicted_stage = stage_encoder.inverse_transform(prediction)[0]

            # Save the prediction to the database
            PredictionStage.objects.create(
                cow_name=cow_name,
                cow_id=cow_id,
                predicted_stage=predicted_stage
            )
    else:
        form = PredictionForm()

    return render(request, 'predictor/predict_theileriosis_stage.html', {'form': form, 'prediction': predicted_stage})

def view_predictions(request):
    predictions = PredictionStage.objects.all()
    return render(request, 'predictor/view_prediction_stages.html', {'predictions': predictions})