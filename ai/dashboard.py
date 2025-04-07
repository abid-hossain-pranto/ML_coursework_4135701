import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Simulated Data and Model (Replace with your actual data/model)
# -----------------------------
# For demonstration purposes, we'll simulate a best model prediction function.
def predict_quality(input_data):
    # Replace with your actual model prediction code
    return np.random.choice([1, 2, 3, 4])

# Simulated production metrics
overall_production_quality = "95%"
product_scrap_rate = "3%"

# Simulated Confusion Matrix for 4 classes
conf_matrix = np.array([[8, 0, 0, 0],
                        [0, 9, 0, 0],
                        [0, 0, 6, 0],
                        [0, 0, 0, 5]])

# Simulated Feature Importance Data
feature_importance_df = pd.DataFrame({
    "Feature": ["Melt temperature", "Mold temperature", "time_to_fill", "Cycle time"],
    "Importance": [0.35, 0.25, 0.20, 0.20]
})

# Simulated Class-wise Evaluation Metrics
class_metrics = pd.DataFrame({
    "Class": [1, 2, 3, 4],
    "Precision": [1.00, 1.00, 1.00, 1.00],
    "Recall": [1.00, 1.00, 1.00, 1.00],
    "F1-score": [1.00, 1.00, 1.00, 1.00]
})

# Simulated ANOVA p-values Data
anova_p_values = pd.DataFrame({
    "Feature": ["Cycle time", "Plasticizing time", "Mold temperature"],
    "p-value": [0.000, 0.000, 0.002]
})

# -----------------------------
# Sidebar - User Input
# -----------------------------
st.sidebar.header("Input Process Parameters")
# Replace with actual ranges from your dataset
melt_temp = st.sidebar.slider("Melt Temperature", min_value=80.0, max_value=160.0, value=106.0)
mold_temp = st.sidebar.slider("Mold Temperature", min_value=75.0, max_value=85.0, value=81.0)
time_to_fill = st.sidebar.slider("Time to Fill", min_value=5.0, max_value=12.0, value=7.5)
cycle_time = st.sidebar.slider("Cycle Time", min_value=70.0, max_value=80.0, value=75.0)

# Create a DataFrame from user inputs
input_data = {
    "Melt temperature": melt_temp,
    "Mold temperature": mold_temp,
    "time_to_fill": time_to_fill,
    "ZUx - Cycle time": cycle_time
    # Add additional parameters as required
}
input_df = pd.DataFrame([input_data])

st.write("### User Input Parameters")
st.write(input_df)

# -----------------------------
# Model Prediction Section
# -----------------------------
st.header("Model Prediction")
# In practice, you would standardize input_df if necessary before prediction.
predicted_quality = predict_quality(input_data)
st.write(f"**Predicted Quality Class:** {predicted_quality}")

# -----------------------------
# Production Metrics Display
# -----------------------------
st.header("Production Metrics")
st.write(f"**Overall Production Quality:** {overall_production_quality}")
st.write(f"**Product Scrap Rate:** {product_scrap_rate}")

# -----------------------------
# Confusion Matrix Display
# -----------------------------
st.header("Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Class 1", "Class 2", "Class 3", "Class 4"],
            yticklabels=["Class 1", "Class 2", "Class 3", "Class 4"],
            ax=ax_cm)
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# -----------------------------
# Feature Importance Display
# -----------------------------
st.header("Feature Importance")
fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
# To avoid FutureWarning, we can explicitly specify hue if desired.
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis", ax=ax_imp)
ax_imp.set_title("Feature Importance")
st.pyplot(fig_imp)

# -----------------------------
# Class-wise Evaluation Metrics
# -----------------------------
st.header("Class-wise Evaluation Metrics")
st.table(class_metrics)

# -----------------------------
# ANOVA Summary Display
# -----------------------------
st.header("ANOVA Summary")
st.write("""
ANOVA analysis indicates that certain process parameters, such as **Cycle Time** and **Plasticizing Time**, 
have statistically significant impacts on product quality. These findings support the inclusion of these features 
in the predictive model.
""")
fig_anova, ax_anova = plt.subplots(figsize=(8, 4))
sns.barplot(x="p-value", y="Feature", data=anova_p_values, palette="coolwarm", ax=ax_anova)
ax_anova.set_title("ANOVA p-values for Key Features")
st.pyplot(fig_anova)
