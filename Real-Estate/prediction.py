import streamlit as st
import numpy as np
import pandas as pd
import pickle
import math
import seaborn as sns
import matplotlib.pyplot as plt
import shap

df = pd.read_csv('binaaz_data.csv')
# Load trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set background color to blue using custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #DFD0B8;
    }
    </style>
    """, unsafe_allow_html=True)


page = st.sidebar.radio('', ['Agenda' ,'Graphics', 'Prediction Page'])
st.sidebar.image("home.jpg")


if page == 'Agenda':
    st.title('Data Analytics with Python')
    st.markdown('<br>' * 2, unsafe_allow_html = True)
    st.header('Dataset: BINA AZ')
    st.markdown('<br>', unsafe_allow_html = True)
    st.subheader('> Preprocessing')
    st.subheader('> Model Building and Optimization')
    st.subheader('> Visualisation')
    st.markdown('<br>' * 3, unsafe_allow_html = True)
    st.subheader('Presenter: Nijat Mammadov')



elif page == 'Graphics':

    st.title('Visual Insights and Charts')

    st.subheader('Distribution of Flat Area')
    fig, ax = plt.subplots()
    sns.histplot(data = df, x = 'area_m2', kde = True, ax=ax)
    st.pyplot(fig)
    
    st.subheader('Distribution of Room Numbers')
    fig, ax = plt.subplots()
    sns.countplot(data = df, x = 'room_number', ax=ax)
    st.pyplot(fig)


    st.subheader('Price Ratio by Repair')
    fig, ax = plt.subplots()
    sns.boxplot(data = df, x = 'repair', y = 'price_azn_1m2', ax=ax)
    st.pyplot(fig)


    location_columns = ['28_May_m.', 'Elmlər_Nizami_Xətai_m.', 'Gənclik_Nərimanov_m.',
                    'Nardaran_AgSheher_q.', 'Nərimanov_r.', 'Sahil_m.',
                    'Səbail_Nəsimi_r.', 'Xətai_r.', 'İçəri_Şəhər_m.']
    
    df_location = df[location_columns + ['price_azn_1m2']]
    df_melted = df_location.melt(id_vars = 'price_azn_1m2', var_name = 'location', value_name = 'is_here')
    df_melted = df_melted[df_melted['is_here']==1]

    location_means = df_melted.groupby('location')['price_azn_1m2'].mean().sort_values(ascending=False)

    st.subheader('Mean Price Ratio by Location')
    fig, ax = plt.subplots()
    location_means.plot(kind='bar')
    st.pyplot(fig)


elif page == 'Prediction Page':

    st.title("Apartment Price Prediction (AZN)")

    # User Inputs
    room_number = st.number_input("Room Number", min_value=1, step=1)
    title_deed = st.selectbox("Has Title Deed? (Çıxarış))", ['No', 'Yes'])
    mortgage = st.selectbox("Eligible for Mortgage? (İpoteka)", ['No', 'Yes'])
    repair = st.selectbox("Repair Condition (Təmirli?))", ['No', 'Yes'])
    apartment_floor = st.number_input("Apartment Floor", min_value=0, step=1)
    building_floor = st.number_input("Total Building Floors", min_value=1, step=1)
    category = st.selectbox("Category (Yeni Tikili?)", ['No', 'Yes'])
    area_m2 = st.number_input("Area (m²)", min_value=1.0)

    title_deed = 1 if title_deed == 'Yes' else 0
    mortgage = 1 if mortgage == 'Yes' else 0
    repair = 1 if repair == 'Yes' else 0
    category = 1 if category == 'Yes' else 0


    # Location Dropdown (single choice)
    location = st.selectbox(
        "Select the closest location:",
        [
            "None",
            "28 May metro",
            "Elmlər metro",
            "Nizami metro",
            "Xətai metro",
            "Gənclik metro",
            "Nərimanov metro",
            "Sahil metro",
            "İçəri Şəhər metro",
            "Nardaran area",
            "Ağ şəhər area",
            "Nərimanov region",
            "Səbail region",
            "Nəsimi region",
            "Xətai region",
        ]
    )

    # Metro and town center
    is_metro = st.selectbox("Close to metro?", ['No', 'Yes'])
    is_town = st.selectbox("Located in Town Center?", ['No', 'Yes'])

    is_metro = 1 if is_metro == 'Yes' else 0
    is_town = 1 if is_town == 'Yes' else 0

    # --- One-hot encoding for location ---
    may_28 = elmler_m = nizami_m = xetai_m = 0
    genclik_m = nerimanov_m = nardaran_q = agseher_q = 0
    nerimanov_r = sahil_m = sebail_r = nesimi_r = xetai_r = icheri_seher_m = 0

    if location == "28 May metro":
        may_28 = 1
    elif location == "Elmlər metro":
        elmler_m = 1
    elif location == "Nizami metro":
        nizami_m = 1
    elif location == "Xətai metro":
        xetai_m = 1
    elif location == "Gənclik metro":
        genclik_m = 1
    elif location == "Nərimanov metro":
        nerimanov_m = 1
    elif location == "Sahil metro":
        sahil_m = 1
    elif location == "İçəri Şəhər metro":
        icheri_seher_m = 1
    elif location == "Nardaran area":
        nardaran_q = 1
    elif location == "Ağ şəhər area":
        agseher_q = 1
    elif location == "Nərimanov region":
        nerimanov_r = 1
    elif location == "Səbail region":
        sebail_r = 1
    elif location == "Nəsimi region":
        nesimi_r = 1
    elif location == "Xətai region":
        xetai_r = 1

    # Derived Features
    total_ratio = room_number / area_m2
    room_number_log = np.log1p(room_number)
    room_number_exp = np.exp(room_number)
    room_number_squared = room_number ** 2

    building_floor_log = np.log1p(building_floor)
    building_floor_exp = np.exp(building_floor)
    building_floor_squared = building_floor ** 2

    apartment_floor_log = np.log1p(apartment_floor)
    apartment_floor_exp = np.exp(apartment_floor)
    apartment_floor_squared = apartment_floor ** 2

    # Prediction trigger
    if st.button("Predict Price"):
        input_data = np.array([[
            room_number, title_deed, mortgage, repair,
            apartment_floor, building_floor, category, area_m2,
            may_28, elmler_m or nizami_m or xetai_m,  # merged as one for model if applicable
            genclik_m or nerimanov_m,
            nardaran_q or agseher_q,
            nerimanov_r, sahil_m,
            sebail_r or nesimi_r, xetai_r, icheri_seher_m,
            is_metro, is_town,
            total_ratio,
            room_number_log, room_number_exp, room_number_squared,
            building_floor_log, building_floor_exp, building_floor_squared,
            apartment_floor_log, apartment_floor_exp, apartment_floor_squared
        ]])

        prediction = model.predict(input_data)
        st.subheader(f"Predicted Apartment Price: {prediction[0]:,.2f} AZN")





# python3 -m pip install lightgbm==4.3.0
# python3 -m pip install --upgrade scikit-learn
# python3 -m streamlit run prediction.py
