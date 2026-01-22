# Food Delivery Time Prediction 🍔🚴‍♂️

## Project Overview
This project is a machine learning–based web application that predicts the estimated food delivery time based on multiple real-world factors such as distance, preparation time, traffic conditions, weather, and order details. The application is built using Streamlit for the frontend and a trained Gradient Boosting Regression model for prediction.

The goal of this project is to demonstrate how data-driven models can improve delivery time estimation in online food delivery platforms.

---

## Features
- Interactive Streamlit dashboard
- Real-time delivery time prediction
- Input validation with user warnings
- Clean and user-friendly UI
- Backend machine learning integration

---

## Input Parameters
The following inputs are taken from the user:

- Distance (km)
- Preparation time (minutes)
- Number of items in the order
- Order hour
- Day of the week
- Traffic density (Low / Medium / High)
- Weather condition (Cloudy / Foggy / Rainy)
- Restaurant type
- Delivery vehicle type
- Peak hour indicator
- Weekend indicator

---

## Validation Logic
The application performs input validation to improve reliability:
- Shows warnings if distance or preparation time is entered as 0
- Prevents prediction when critical inputs are invalid
- Displays alerts for unusually high distance or preparation time values

---

## Machine Learning Model
- **Model Used:** Gradient Boosting Regressor
- **Preprocessing:** Feature encoding and scaling using a pre-trained scaler
- **Output:** Estimated delivery time in minutes

The model is trained offline and loaded into the Streamlit app using `.pkl` files.

---

## Technology Stack
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## How to Run the Project Locally

1. Clone the repository:
2. Install dependencies: pip install -r requirements.txt
3. Run the Streamlit app:streamlit run app.py


---

## Conclusion
This project demonstrates the practical application of machine learning in predicting delivery times and highlights the integration of ML models with interactive web applications using Streamlit.


