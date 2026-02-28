import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@400;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto Slab', serif;
}

.stApp {
    background: radial-gradient(circle at 5% 5%, #1f2c2e 0%, #0f1718 100%);
    color: #E2E8F0;
}

#MainMenu, footer, header {visibility: hidden;}

/* Header / Title */
.app-title {
    font-family: Verdana, Helvetica, sans-serif;
    font-weight: 600;
    font-size: 2rem;
    letter-spacing: 0.1px;
    color: #b7c3c7;
    
}

.app-subtitle {
    font-family: 'Roboto Slab', serif;
    font-weight: 400;
    font-size: 1rem;
    letter-spacing: 1px;
    color: #8fb3b0;
}

/* Section labels */
.section-label {
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8fb3b0;
    margin-bottom: 10px;
}

/* Metric Box */
.metric-box {
    text-align: center;
    padding: 40px;
    border-radius: 20px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}

/* Risk Colors */
.high-risk {
    color: #FF4B4B;
    text-shadow: 0 0 20px rgba(255,75,75,0.4);
}
.low-risk {
    color: #00D1B2;
    text-shadow: 0 0 20px rgba(0,209,178,0.4);
}

/* Dark Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
textarea, input {
    background-color: #0e1117 !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 6px;
}

div[data-baseweb="slider"] > div {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL & ARTIFACTS ----------------
@st.cache_resource
def load_engine():
    model = load_model('model.h5')
    with open('onehot_encoder_geo.pkl','rb') as f:
        onehot_encoder_geo = pickle.load(f)
    with open('label_encoder_gender.pkl','rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    return model, onehot_encoder_geo, label_encoder_gender, scaler

model, onehot_encoder_geo, label_encoder_gender, scaler = load_engine()

# ---------------- HEADER ----------------
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<h1 class='app-title'>Customer Churn Engine</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<p class='app-subtitle' style='text-align:right;'>ANN Engine v1.0</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- INPUT GRID ----------------
c1, c2, c3 = st.columns(3, gap="large")

# --- Column 1 ---
with c1:
    st.markdown("<div class='section-label'>Demographics</div>", unsafe_allow_html=True)
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", 18, 92, 35)

# --- Column 2 ---
with c2:
    st.markdown("<div class='section-label'>Financial Profile</div>", unsafe_allow_html=True)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    balance = st.number_input("Balance", value=50000.0)
    estimated_salary = st.number_input("Estimated Salary", value=70000.0)

# --- Column 3 ---
with c3:
    st.markdown("<div class='section-label'>Engagement</div>", unsafe_allow_html=True)
    tenure = st.slider("Tenure", 0, 10, 5)
    num_of_products = st.slider("Number of Products", 1, 4, 2)

    sub1, sub2 = st.columns(2)
    with sub1:
        has_cr_card = st.checkbox("Has Credit Card", value=False)
    with sub2:
        is_active_member = st.checkbox("Active Member", value=False)

# ---------------- DATA PROCESSING ----------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card else 0],
    'IsActiveMember': [1 if is_active_member else 0],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

final_input = scaler.transform(
    pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
)

prediction = float(model.predict(final_input)[0][0])

# ---------------- OUTPUT SECTION ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
left, center, right = st.columns([1,2,1])

with center:
    status_class = "high-risk" if prediction > 0.5 else "low-risk"
    status_text = "LIKELY TO CHURN" if prediction > 0.5 else "CUSTOMER RETAINED"

    st.markdown(f"""
        <div class="metric-box">
            <div class="section-label">Churn Probability</div>
            <h1 class="{status_class}" style="font-size:4rem;">{prediction:.2%}</h1>
            <p style="letter-spacing:3px; font-weight:600;">{status_text}</p>
        </div>
    """, unsafe_allow_html=True)

# subtle progress bar
st.markdown(f"""
<div style="width:100%; height:3px; background:rgba(255,255,255,0.05); margin-top:50px;">
    <div style="width:{prediction*100}%; height:3px; 
    background:{'#FF4B4B' if prediction > 0.5 else '#00D1B2'};"></div>
</div>
""", unsafe_allow_html=True)