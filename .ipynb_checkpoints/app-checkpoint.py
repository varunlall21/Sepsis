import streamlit as st
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier

# === Load model ===
tabnet = TabNetClassifier()
tabnet.load_model("tabnet_optimized.zip")

# === Load dataset and extract features ===
df = pd.read_csv("ProcessedDataset.csv")
all_features = list(df.drop(columns=["SepsisLabel"]).columns)

# Define key vitals features (for quick editing)
vitals = [
    'HR', 'SBP', 'MAP', 'Resp', 'Temp', 'WBC', 'Lactate', 
    'Creatinine', 'Platelets', 'BUN', 'Bilirubin_total', 'FiO2', 'O2Sat'
]

# === Select real septic patient (SepsisLabel == 1) as base example ===
septic_row = df[df["SepsisLabel"] == 1].iloc[0]

# Convert septic_row to a dictionary for defaults
default_input = septic_row[all_features].to_dict()

# === Streamlit UI Layout ===

st.set_page_config(page_title="Sepsis Detection Dashboard", layout="wide")
st.title("🧠 Sepsis Detection Dashboard")
st.markdown("""
This dashboard uses a real septic patient example as the base.
You can adjust the patient’s values below and run a prediction.
""")

with st.form(key="patient_input_form"):
    st.subheader("Key Vitals")
    # Layout key vitals in two columns
    col1, col2 = st.columns(2)
    user_input = {}
    for i, feature in enumerate(vitals):
        default_val = default_input.get(feature, 0.0)
        # Send half the features to col1 and the rest to col2
        if i < len(vitals) // 2:
            user_input[feature] = col1.number_input(f"{feature}", value=float(default_val))
        else:
            user_input[feature] = col2.number_input(f"{feature}", value=float(default_val))
            
    # Expander for other features
    remaining_features = [feat for feat in all_features if feat not in vitals]
    with st.expander("Other Features (Optional)"):
        for feature in remaining_features:
            user_input[feature] = st.number_input(
                f"{feature}", value=float(default_input.get(feature, 0.0))
            )
    
    # Submit button for the form
    submitted = st.form_submit_button("Run Sepsis Prediction")

# Prepare the full input vector using the order in all_features
if submitted:
    # Create a DataFrame for the model prediction.
    # The model expects the features in the same order as in all_features.
    input_df = pd.DataFrame([[user_input[feat] for feat in all_features]], columns=all_features)
    
    # Optional: Show the full input for debugging
    st.subheader("Input Features for Prediction")
    st.dataframe(input_df)
    
    # Run prediction
    pred_class = tabnet.predict(input_df.values)[0]
    pred_prob = tabnet.predict_proba(input_df.values)[0][1]
    
    # Option to adjust threshold (for example, use custom threshold 0.3)
    custom_threshold = 0.5
    st.subheader("Prediction Result")
    if pred_prob > custom_threshold:
        st.error(f"🔴 Sepsis Detected! Probability: {pred_prob:.2f}")
    else:
        st.success(f"🟢 No Sepsis Detected. Probability: {pred_prob:.2f}")
