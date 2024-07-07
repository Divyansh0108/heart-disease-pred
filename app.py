import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Load the data
df = pd.read_csv('heart_disease_data.csv')

# Prepare the data
X = df.drop('target', axis=1)
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_pred, Y_test)
    results[name] = accuracy
    print(f"{name} Test Accuracy: {accuracy}")

# Determine the best model based on test accuracy
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with Test Accuracy: {results[best_model_name]}")

# Example input data for prediction
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
input_data = (66, 0, 3, 150, 226, 0, 1, 114, 0, 2.6, 0, 0, 2)
np_df = np.asarray(input_data).reshape(1, -1)

# Convert to DataFrame with correct feature names
df_input = pd.DataFrame(np_df, columns=feature_names)

# Predict using the best model
prediction = best_model.predict(df_input)

if prediction[0] == 0:
    print("This person doesn't have heart disease!")
else:
    print("This person has heart disease!")

# Accuracy on training data
X_train_prediction = best_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# Accuracy on test data
X_test_prediction = best_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print(f"Training Data Accuracy: {training_data_accuracy}")
print(f"Test Data Accuracy: {test_data_accuracy}")

# Streamlit application
st.title('Heart Disease Prediction Model')

input_text = st.text_input('Provide comma separated features to predict heart disease')
sprted_input = input_text.split(',')
img = Image.open('heartImage.jpg')
st.image(img, width=150)

try:
    np_df = np.asarray(sprted_input, dtype=float)
    reshaped_df = np_df.reshape(1, -1)
    prediction = best_model.predict(reshaped_df)
    if prediction[0] == 0:
        st.write("This person doesn't have heart disease")
    else:
        st.write("This person has heart disease")
except ValueError:
    st.write('Please provide comma separated values')

st.subheader("About Data")
st.write(df)
st.subheader("Model Performance on Training Data")
st.write(training_data_accuracy)
st.subheader("Model Performance on Test Data")
st.write(test_data_accuracy)
