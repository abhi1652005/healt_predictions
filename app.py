from pathlib import Path
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Healthcare Cost Prediction App", page_icon="🏥", layout="centered"
)

# Define the absolute path to the model file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"


@st.cache_resource
def load_model():
  """Loads the trained machine learning model."""
  with open(MODEL_PATH, "rb") as f:
    return pickle.load(f)


# Load model safely
try:
  model = load_model()
except FileNotFoundError:
  st.error(
      f"Model file not found at {MODEL_PATH}. Please ensure 'model.pkl' is"
      " uploaded to your repository."
  )
  st.stop()

# App UI Header
st.title("🏥 Healthcare Cost Prediction")
st.write(
    "Enter your details below to predict expected annual medical costs using"
    " your machine learning model."
)

# Input fields based on your model features
# (Age, BMI, and Daily Steps as examples based on common healthcare datasets)
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
daily_steps = st.number_input(
    "Daily Steps", min_value=0, max_value=50000, value=5000
)

# Prediction button
if st.button("Predict Cost"):
  # Format input matching your model's expected feature dimensions
  # Update this list/array structure to match all the features your model expects
  input_data = [[age, bmi, daily_steps]]  # Adjust based on exact feature count

  try:
    prediction = model.predict(input_data)
    st.success(
        f"Estimated Annual Medical Cost: 💰 ${prediction[0]:,.2f}"
    )  #[cite: 1]
  except Exception as e:
    st.error(
        "An error occurred during prediction. Ensure the number of input"
        f" features matches your model.\nDetails: {e}"
    )
