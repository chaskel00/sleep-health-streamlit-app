import streamlit as st
import pandas as pd
import joblib


# load the saved model
model_info = joblib.load("sleep_model.pkl")

model = model_info["model"]
scaler = model_info["scaler"]
rfe = model_info["rfe"]
columns = model_info["columns"]


st.title("Sleep Disorder Prediction")

st.caption("Built with Python, Scikit-Learn, and Streamlit")

st.markdown("""
This application predicts the most likely sleep disorder based on
health and lifestyle information.

The model was trained using the **Sleep Health and Lifestyle Dataset**
from Kaggle and a **Random Forest classifier**.
""")

with st.expander("About this project"):
    st.write("""
This project was originally completed for my Statistical Learning course
at Indiana University Indianapolis.

The Random Forest model achieved approximately **97% accuracy**
when classifying:

- None
- Insomnia
- Sleep Apnea

The model uses features including sleep duration, BMI category,
and blood pressure to make predictions.
""")

st.write("Enter the information below and click predict.")


# user inputs
age = st.number_input("Age", min_value=18, max_value=90, value=30)

sleep_duration = st.number_input("Sleep Duration", min_value=0.0, max_value=12.0, value=7.0, step=0.1)

quality_of_sleep = st.slider("Quality of Sleep", min_value=1, max_value=10, value=7)

physical_activity = st.slider("Physical Activity Level", min_value=0, max_value=100, value=50)

stress_level = st.slider("Stress Level", min_value=1, max_value=10, value=5)

bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])

heart_rate = st.number_input("Heart Rate", min_value=40, max_value=140, value=75)

daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=7000)

systolic = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)

diastolic = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=130, value=80)

gender = st.selectbox("Gender", ["Male", "Female"])

occupation = st.selectbox(
    "Occupation",
    [
        "Accountant",
        "Doctor",
        "Engineer",
        "Lawyer",
        "Manager",
        "Nurse",
        "Sales Representative",
        "Salesperson",
        "Scientist",
        "Software Engineer",
        "Teacher"
    ]
)


if st.button("Predict Sleep Disorder"):

    bmi_map = {
        "Normal": 0,
        "Overweight": 1,
        "Obese": 2
    }

    # start with every model column set to 0
    input_data = pd.DataFrame([[0] * len(columns)], columns=columns)

    # fill in the normal numeric columns
    input_data["Age"] = age
    input_data["Sleep Duration"] = sleep_duration
    input_data["Quality of Sleep"] = quality_of_sleep
    input_data["Physical Activity Level"] = physical_activity
    input_data["Stress Level"] = stress_level
    input_data["BMI Category"] = bmi_map[bmi_category]
    input_data["Heart Rate"] = heart_rate
    input_data["Daily Steps"] = daily_steps
    input_data["Systolic"] = systolic
    input_data["Diastolic"] = diastolic

    # match the one hot encoded columns from training
    gender_col = "Gender_" + gender
    occupation_col = "Occupation_" + occupation

    if gender_col in input_data.columns:
        input_data[gender_col] = 1

    if occupation_col in input_data.columns:
        input_data[occupation_col] = 1

    # apply the same scaler and feature selection from training
    input_scaled = scaler.transform(input_data)
    input_final = rfe.transform(input_scaled)

    prediction = model.predict(input_final)[0]
    probabilities = model.predict_proba(input_final)[0]

    st.subheader("Prediction")
    st.success(prediction)

    st.subheader("Prediction Confidence")

    prob_df = pd.DataFrame({
        "Sleep Disorder": model.classes_,
        "Probability": probabilities
    })

    prob_df = prob_df.sort_values(
        by="Probability",
        ascending=False
    )

    st.bar_chart(
        prob_df.set_index("Sleep Disorder")
    )