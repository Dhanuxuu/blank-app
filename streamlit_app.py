# import streamlit as st

# st.title("🎈 My new Streamlit app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('salary_model.pkl')

# User input
st.title("Employee Salary Predictor")
age = st.number_input("Age", min_value=18, max_value=65)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
job_title = st.text_input("Job Title")
experience = st.number_input("Years of Experience", min_value=0.0)

# Preprocess input
input_df = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'education': [education],
    'job_title': [job_title],
    'experience': [experience]
})

# Predict
if st.button("Predict Salary"):
    # You may need to apply encoding/scaling here
    salary = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ${salary:,.2f}")
