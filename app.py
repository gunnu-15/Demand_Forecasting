import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Don't load them again & again, instead CACHE it
@st.cache_resource

# Start loading the Encoders & Model
def load_artifacts():
        with open('xgboost_demand_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)

        return model,encoders

model, label_encoders = load_artifacts()

st.title('Demand Forecasting App')
st.divider()
st.header('Input Features')

price = st.number_input('Price', min_value=0.0, value=50.0)
discount = st.number_input('Discount', min_value=0, max_value=100, value=10)
inventory_level = st.number_input('Inventory Level', min_value=0, value=100)
promotion = st.selectbox('Promotion', [0,1])
competitor_pricing = st.number_input('Competitor Pricing', min_value=0.0, value=50.0)

# We'll show classes of Category - Electronics, Toys, Furniture, etc this way
category = st.selectbox(
    'Category',
    label_encoders['Category'].classes_.tolist()
)

input_data = pd.DataFrame({
    'Price': [price],
    'Discount': [discount],
    'Inventory Level': [inventory_level],
    'Promotion': [promotion],
    'Competitor Pricing': [competitor_pricing],
    'Category': [category]
})

# Encode Categorical Features using for loop. For each column, we're going to do this
# operation in labeling encoder items
for col, encoder in label_encoders.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])

# Create a Prediction Button
st.divider()
if st.button("Predict Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Demand: {int(prediction)} Units')