import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats  # Import for ANOVA

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("CW_Dataset_4135701.csv")
except FileNotFoundError:
    st.error("Dataset file 'CW_Dataset_4135701.csv' not found. Please ensure the file is in the working directory.")
    st.stop()

# -----------------------------
# Preprocess the Data (Split features and target)
# -----------------------------
target_column = df.columns[-1]
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Feature Scaling (For SVM, KNN, ANN)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Initialize Models
# -----------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu', solver='adam', alpha=0.0001, max_iter=1000, random_state=42)
}

# -----------------------------
# ANOVA Summary Function
# -----------------------------
def anova_summary(df, features, target_column):
    p_values = []
    for feature in features:
        # Grouping data by the target column for ANOVA test
        groups = [df[df[target_column] == i][feature] for i in df[target_column].unique()]
        _, p_value = stats.f_oneway(*groups)
        p_values.append(p_value)
    return pd.DataFrame({"Feature": features, "p-value": p_values})

# -----------------------------
# Cross-validation and Model Training
# -----------------------------
st.header("Model Cross-Validation Scores")
cv_results = []
for name, model in models.items():
    X_train_input = X_train_scaled if name in ["SVM", "KNN", "ANN"] else X_train
    scores = cross_val_score(model, X_train_input, y_train, cv=10, scoring='accuracy')
    cv_results.append((name, scores.mean(), scores.std()))
    st.write(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

# -----------------------------
# Train Models on Full Data and Evaluate on Test Set
# -----------------------------
st.header("Model Evaluation on Test Set")
classification_reports = {}
model_accuracies = []

# Train all models and store their evaluation reports
for model_name, model in models.items():
    X_train_input = X_train_scaled if model_name in ["SVM", "KNN", "ANN"] else X_train
    X_test_input = X_test_scaled if model_name in ["SVM", "KNN", "ANN"] else X_test
    
    model.fit(X_train_input, y_train)  # Fit the model to the training data
    y_pred = model.predict(X_test_input)
    
    classification_reports[model_name] = classification_report(y_test, y_pred)
    model_accuracies.append((model_name, model.score(X_test_input, y_test)))
    
    st.subheader(f"{model_name} Classification Report:")
    st.text(classification_reports[model_name])

# -----------------------------
# Model Performance Visualization (Accuracy Comparison)
# -----------------------------
st.header("Model Accuracy Comparison")

# Extract model names and their corresponding accuracies
model_names = [item[0] for item in model_accuracies]
accuracies = [item[1] for item in model_accuracies]

# Create a bar plot for model accuracy comparison
fig = plt.figure(figsize=(10, 6))
plt.barh(model_names, accuracies, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
st.pyplot(fig)

# -----------------------------
# Confusion Matrix Visualization
# -----------------------------
st.header("Confusion Matrix for Each Model")

# Plot confusion matrix for each model
for model_name, model in models.items():
    X_test_input = X_test_scaled if model_name in ["SVM", "KNN", "ANN"] else X_test
    y_pred = model.predict(X_test_input)
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f"Confusion Matrix for {model_name}")
    st.pyplot(fig)

# -----------------------------
# Feature Importance Visualization (For Random Forest)
# -----------------------------
st.header("Feature Importance (Random Forest)")

# Extract feature importance from Random Forest model
rf_model = models["Random Forest"]
feature_importances = rf_model.feature_importances_
features = X.columns

# Create a bar plot for feature importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features, palette='viridis')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance (Random Forest)')
st.pyplot(fig)

# -----------------------------
# ANOVA Summary Section
# -----------------------------
st.header("ANOVA Summary")

# Perform ANOVA on process parameters (excluding target column)
anova_results = anova_summary(df, X.columns, target_column)

# Display ANOVA Summary
st.write("""
ANOVA analysis indicates that certain process parameters have statistically significant impacts on product quality.
""")
st.table(anova_results)

# Visualization of ANOVA p-values
fig_anova, ax_anova = plt.subplots(figsize=(8, 4))
sns.barplot(x="p-value", y="Feature", data=anova_results, palette="coolwarm", ax=ax_anova)
ax_anova.set_title("ANOVA p-values for Key Features")
ax_anova.set_xlabel("p-value")
ax_anova.set_ylabel("Feature")
st.pyplot(fig_anova)

# -----------------------------
# Sidebar for User Input
# -----------------------------
st.sidebar.header("Input Process Parameter Values")

# Let the user choose process parameters via the sidebar
input_data = {}
for column in X.columns:
    min_val = float(df[column].min())
    max_val = float(df[column].max())
    default_val = float(df[column].mean())
    input_data[column] = st.sidebar.slider(column, min_value=min_val, max_value=max_val, value=default_val)

input_df = pd.DataFrame([input_data])

# Show the user inputs
st.write("### User Input Parameters")
st.write(input_df)

# -----------------------------
# Predict Using the Trained Model (e.g., SVM)
# -----------------------------
def predict_quality(input_data):
    # Apply feature scaling to the input data if necessary (for SVM, KNN, ANN)
    input_data_scaled = scaler.transform(input_data)
    
    # Let's predict using the SVM model (can be changed to any model)
    model = models["SVM"]  # You can choose any model here from the `models` dictionary
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Make prediction
predicted_quality = predict_quality(input_df)
st.header("Model Prediction")
st.write(f"**Predicted Quality Class:** {predicted_quality}")
