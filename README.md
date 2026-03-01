# Customer Churn Classification (ANN)

This project presents a production-ready Customer Churn Prediction system built using an Artificial Neural Network (ANN) implemented in TensorFlow and deployed through Streamlit.

The system predicts the probability of customer attrition based on demographic, financial, and engagement-related features. It includes end-to-end model development, preprocessing pipelines, serialized artifacts, and a fully interactive web application.

##  Live Demo
[Churn Classifier](https://customerchurnpredictionann-4.streamlit.app/)

---

## Problem Statement

Customer churn is a critical business challenge in the banking and financial services sector. Identifying customers at risk of leaving enables proactive retention strategies, reducing revenue loss and improving customer lifetime value.

This project aims to:

- Build a robust binary classification model
- Ensure preprocessing consistency between training and inference
- Deliver a deployment-ready, interactive prediction interface
- Maintain reproducibility and clean project architecture

---

## Dataset

**File:** `Churn_Modelling.csv`

The dataset contains structured customer-level banking data, including:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited (Target Variable)

**Target Variable:**
- `1` → Customer churned
- `0` → Customer retained

---

## 4. Project Architecture
CHURN CLASSIFICATION ANN/   
│   
├── app.py   
├── Churn_Modelling.csv   
├── experiments.ipynb   
├── prediction.ipynb   
│   
├── model.h5   
├── scaler.pkl   
├── label_encoder_gender.pkl   
├── onehot_encoder_geo.pkl   
│     
├── requirements.txt      
└── runtime.txt   



---

### Feature Engineering

- Label Encoding (Gender)
- One-Hot Encoding (Geography)
- Standard Scaling (Numerical Features)

Preprocessing artifacts were serialized to ensure consistency during inference.

Saved artifacts:
- `label_encoder_gender.pkl`
- `onehot_encoder_geo.pkl`
- `scaler.pkl`

---

### Model Architecture

The model is a fully connected Artificial Neural Network implemented using TensorFlow.

**Architecture Overview:**
- Input layer (scaled and encoded features)
- Hidden dense layers with nonlinear activation
- Output layer with sigmoid activation

**Output:**
- A probability score between 0 and 1 representing churn likelihood.

Model file: model.h5


---

## Streamlit Application

**File:** `app.py`

The application provides:

- Structured input interface (Demographics, Financial Profile, Engagement)
- Real-time preprocessing
- Live ANN inference
- Probability visualization
- Risk classification indicator

### Prediction Logic
- Probability > 0.5 → Likely to Churn
- Probability ≤ 0.5 → Customer Retained

The application uses `@st.cache_resource` to optimize model loading performance.

---

## Installation & Local Setup

-  Recommended Python Version - Python 3.11

## **How to Run Locally**

**Clone the repository:**
```bash
git clone https://github.com/jotisukheja/Customer_Churn_Prediction_Ann.git
cd Customer_Churn_Prediction_Ann
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the app::**
```bash
streamlit run app.py
```

## Technical Stack
- python-3.11 (runtime.txt) 
- TensorFlow 
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Pickle (Model Serialization)
