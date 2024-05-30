#!pip install tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from joblib import parallel_backend
import joblib
from tqdm import tqdm

# Load the dataset
file_path = 'theileriosis_synthetic_dataset.csv'
theileriosis_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(theileriosis_data.head())

# Display the dataset structure
print(theileriosis_data.info())

# Check for missing values
print(theileriosis_data.isnull().sum())

# Handle missing values if any (e.g., by filling with the mean or median)
numeric_columns = theileriosis_data.select_dtypes(include=['float64', 'int64']).columns
theileriosis_data[numeric_columns] = theileriosis_data[numeric_columns].fillna(theileriosis_data[numeric_columns].mean())

# Assuming the last column is the target variable 'theileriosis' and the rest are features
X = theileriosis_data.drop(columns=['theileriosis'])
y = theileriosis_data['theileriosis']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Check the class distribution
print("Class distribution after SMOTE:", np.bincount(y_resampled))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a RandomForest model to get feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances and select the top features
feature_importances = rf.feature_importances_
important_features = np.argsort(feature_importances)[::-1][:20]  # Select top 20 features
important_feature_names = X.columns[important_features]

# Select the top features from the DataFrame
X_train = X_train[important_feature_names]
X_test = X_test[important_feature_names]

# Define a function to create a model pipeline with scaling and training
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

# Define individual models
models = [
    ('RandomForest', RandomForestClassifier(random_state=42)),
    ('GradientBoosting', GradientBoostingClassifier(random_state=42)),
    ('SVM', SVC(probability=True, random_state=42)),
    ('KNeighbors', KNeighborsClassifier())
]

# Create the VotingClassifier
voting_clf = VotingClassifier(estimators=models, voting='soft')

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', voting_clf)
])

# Define hyperparameters for GridSearch
param_grid = {
    'classifier__RandomForest__n_estimators': [50, 100, 200],
    'classifier__RandomForest__max_depth': [None, 10, 20, 30],
    'classifier__RandomForest__min_samples_split': [2, 5, 10],
    'classifier__GradientBoosting__n_estimators': [50, 100, 200],
    'classifier__GradientBoosting__learning_rate': [0.01, 0.1, 0.2],
    'classifier__GradientBoosting__max_depth': [3, 5, 7],
    'classifier__SVM__C': [0.1, 1, 10],
    'classifier__SVM__kernel': ['linear', 'rbf'],
    'classifier__KNeighbors__n_neighbors': [3, 5, 7, 9],
    'classifier__KNeighbors__weights': ['uniform', 'distance']
}

# Perform GridSearch with VotingClassifier
with parallel_backend('threading'):
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=10)
    grid_search.fit(X_train, y_train)

# Evaluate the model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"VotingClassifier Accuracy: {accuracy}")

print(f"Best Model Accuracy: {accuracy}")

# Save the best model
model_path = 'theileriosis_model.pkl'
joblib.dump(best_model, model_path)

# Print classification report
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

