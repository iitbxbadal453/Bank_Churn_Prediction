import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


def build_features(credit_score, geography, gender, age, tenure, balance,
                    num_of_products, has_cr_card, is_active_member, estimated_salary):
    # same feature engineering as the training notebook
    balance_salary_ratio = balance / (estimated_salary + 1)
    credit_score_per_age = credit_score / age
    tenure_by_age = tenure / age
    is_zero_balance = 1 if balance == 0 else 0
    inactive_with_products = 1 if (is_active_member == 0 and num_of_products >= 2) else 0
    senior_customer = 1 if age >= 50 else 0
    products_per_tenure = num_of_products / (tenure + 1)

    return {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'BalanceSalaryRatio': balance_salary_ratio,
        'CreditScorePerAge': credit_score_per_age,
        'TenureByAge': tenure_by_age,
        'IsZeroBalance': is_zero_balance,
        'InactiveWithProducts': inactive_with_products,
        'SeniorCustomer': senior_customer,
        'ProductsPerTenure': products_per_tenure,
        'Geography_Germany': 1 if geography == 'Germany' else 0,
        'Geography_Spain': 1 if geography == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
    }


st.title("Bank Customer Churn Predictor")

credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

if st.button('Predict'):

    # 1. build raw + engineered features
    features = build_features(
        credit_score, geography, gender, age, tenure, balance,
        num_of_products, 1 if has_cr_card == "Yes" else 0,
        1 if is_active_member == "Yes" else 0, estimated_salary
    )

    # 2. order columns exactly as the model was trained on
    feature_order = list(getattr(model, 'feature_names_in_', features.keys()))
    input_df = pd.DataFrame([features])[feature_order]

    # 3. scale the same numeric columns the scaler was fit on
    scale_cols = list(getattr(scaler, 'feature_names_in_', []))
    input_scaled = input_df.copy()
    if scale_cols:
        input_scaled[scale_cols] = scaler.transform(input_df[scale_cols])

    # 4. predict
    result = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1] if hasattr(model, 'predict_proba') else None

    # 5. Display
    if result == 1:
        st.header("Likely to Churn")
    else:
        st.header("Likely to Stay")

    if proba is not None:
        st.write(f"Churn probability: {proba:.1%}")