# import streamlit as st

# st.title("🎈 My new Streamlit app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

import streamlit as st
import joblib
import pandas as pd

# Load model and dummy columns used in training
model = joblib.load('model.pkl')
dummy_columns = joblib.load('dummy_columns.pkl')  # You should save this during training

# User input
st.title("Employee Salary Predictor")
age = st.number_input("Age", min_value=18, max_value=65)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
job_title = st.text_input("Job Title")
experience = st.number_input("Years of Experience", min_value=0.0)

# Input to dataframe
input_df = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'education': [education],
    'job_title': [job_title],
    'experience': [experience]
})

# One-hot encode using same training logic
input_df = pd.get_dummies(input_df)

# Add any missing columns that existed in training
for col in dummy_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[dummy_columns]

# Predict
if st.button("Predict Salary"):
    salary = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ${salary:,.2f}")
