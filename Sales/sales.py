import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
#with open('sales_model.pkl', 'rb') as f:
#    model = pickle.load(f)

with open('sales.pkl', 'rb') as file:
    model = pickle.load(file)

# Load data from CSV file
df = pd.read_csv('sales_pbi.csv')


st.title('Total Sales Prediction')  

# Step 1: Select Shop
shop = df['mağaza'].unique()
selected_shop = st.selectbox('Select Shop', shop)

# Step 2: Filter categories based on selected shop
filtered_categories = df[df['mağaza'] == selected_shop]['category'].unique()
selected_category = st.selectbox('Select Category', filtered_categories)

# Step 3: Filter product names based on selected category (and shop)
filtered_product = df[
    (df['mağaza'] == selected_shop) &
    (df['category'] == selected_category)
    ]['product_name'].unique()
selected_product = st.selectbox('Select Product', filtered_product)

# Count input
#count = st.number_input('Enter Count', min_value = 1, step = 1)

def preprocess_features(df):
    categ_cols = df.columns #['mağaza', 'category', 'product_name', 'məhsul_nomresi', '']
    input_encoded = pd.get_dummies(df, columns=categ_cols)

    # Get expected features from the model
    try:
        expected_features = model.feature_names_in_
    except AttributeError:
        st.error('Model does not have "feature_names_in" attribute. Please check your model.')

        return None

    for col in expected_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[expected_features]

    return input_encoded


if st.button('Predict Sales'):
    
    df = pd.DataFrame({
        'mağaza': [selected_shop],
        'category': [selected_category],
        'product_name': [selected_product],
    })


    input_features = preprocess_features(df)

    if input_features is not None:
        st.write('Input features after preprocessing:')
        st.dataframe(input_features)

    predicted_sales = model.predict(input_features)

    st.success(f'Predicted Sales: {predicted_sales[0]:.2f}')

