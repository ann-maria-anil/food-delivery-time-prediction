import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="centered"
)

# -------------------------------
# STYLING
# -------------------------------
st.markdown("""
    <style>
    .main {background-color: #ffe5e5;}  /* Very light red/pink background */
    h1 {color: #ff4b4b; font-family: 'Arial', sans-serif;}
    h2 {color: #ff7f50; font-family: 'Arial', sans-serif;}
    .stButton>button {background-color: #ff4b4b; color: white; font-size: 16px; border-radius:10px;}
    .stNumberInput>div>input {border: 2px solid #ff7f50; border-radius: 8px;}
    .stSelectbox>div>div {border: 2px solid #ff7f50; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("🍔 Food Delivery Time Predictor")
st.markdown("Predict how long your food will take to reach you!")

# -------------------------------
# LOAD MODEL & DATA
# -------------------------------
try:
    model = joblib.load("delivery_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("food_delivery_data.csv")
except Exception as e:
    st.error("❌ Required files not found")
    st.stop()

FEATURE_COLUMNS = list(scaler.feature_names_in_)

# -------------------------------
# USER INPUTS
# -------------------------------
st.markdown("### 📥 Enter Order Details")
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        distance = st.number_input("Distance (km) 🚗", min_value=0.0, step=0.5)
        prep_time = st.number_input("Preparation Time (minutes) 🍳", min_value=0)
        order_size = st.number_input("Order Size (items) 🛒", min_value=1)

        restaurant_type = st.selectbox(
            "Restaurant Type 🍴",
            ["Indian", "Chinese", "Italian", "Fast Food", "Dessert"]
        )

    with col2:
        vehicle_type = st.selectbox(
            "Delivery Vehicle 🛵",
            ["Scooter", "Car"]
        )
        weather_condition = st.selectbox(
            "Weather Condition 🌦",
            ["Cloudy", "Foggy", "Rainy"]
        )
        traffic_level = st.selectbox(
            "Traffic Level 🚦",
            ["Low", "Medium", "High"]
        )
        is_peak_hour = st.checkbox("Is it Peak Hour? ⏰")
        is_weekend = st.checkbox("Is it Weekend? 🗓")

        order_hour = st.number_input("Order Hour (0-23) 🕒", min_value=0, max_value=23)
        day_of_week = st.selectbox(
            "Day of the Week 📅",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        )

# -------------------------------
# INPUT VALIDATION
# -------------------------------
warnings = []

# Distance & Preparation Time
invalid_input = False
if distance == 0:
    warnings.append("⚠️ Distance entered is 0 km!")
    invalid_input = True
if prep_time == 0:
    warnings.append("⚠️ Preparation time entered is 0 minutes!")
    invalid_input = True

# High / unusual values
if distance > 50:
    warnings.append("⚠️ Distance is more than 50 km — prediction may be inaccurate!")
if distance > 100:
    warnings.append("ℹ️ Distance is unusually high — prediction may be less accurate.")
if prep_time > 180:
    warnings.append("ℹ️ Preparation time is unusually high — prediction may be less accurate.")

# Show all warnings to user
for w in warnings:
    st.warning(w)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
st.markdown("---")

if invalid_input:
    st.error("❌ Cannot predict with invalid inputs! Please correct the above warnings.")
else:
    if st.button("🚀 Predict Delivery Time"):
        try:
            # --------- Initialize input ---------
            input_df = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
            input_df['distance_km'] = distance
            input_df['preparation_time'] = prep_time
            input_df['items_count'] = order_size
            input_df['order_hour'] = order_hour

            # Map day to numeric
            day_map = {
                "Monday": 1,
                "Tuesday": 2,
                "Wednesday": 3,
                "Thursday": 4,
                "Friday": 5,
                "Saturday": 6,
                "Sunday": 7
            }
            input_df['day_of_week'] = day_map[day_of_week]

            # Boolean features
            input_df['is_peak_hour'] = int(is_peak_hour)
            input_df['is_weekend'] = int(is_weekend)

            # One-hot features
            rest_col = f"restaurant_type_{restaurant_type}"
            if rest_col in input_df.columns:
                input_df[rest_col] = 1
            vehicle_col = f"vehicle_type_{vehicle_type}"
            if vehicle_col in input_df.columns:
                input_df[vehicle_col] = 1
            weather_col = f"weather_condition_{weather_condition}"
            if weather_col in input_df.columns:
                input_df[weather_col] = 1

            # Traffic numeric
            traffic_map = {"Low": 1, "Medium": 2, "High": 3}
            input_df['traffic_density'] = traffic_map[traffic_level]

            # --------- Scale & predict ---------
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            # --------- OUTPUT - nice card style ---------
            st.markdown(
                f"""
                <div style='background-color:#ffd6d6; padding:20px; border-radius:15px;'>
                <h2 style='color:#ff4b4b;'>🍽 Estimated Delivery Time</h2>
                <h1 style='color:#ff7f50;'>{prediction:.2f} minutes</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error("❌ Prediction failed due to feature mismatch or invalid input")
            st.exception(e)
