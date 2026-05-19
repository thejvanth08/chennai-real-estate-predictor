import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Chennai Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("chennai_house_price_model.pkl")
    return model

model = load_model()

# --------------------------------
# HEADER
# --------------------------------
st.title("🏠 Chennai Real Estate Price Predictor")
st.markdown(
    """
    Predict approximate residential/commercial property prices in Chennai
    using a Machine Learning model trained on historical housing sales data.
    """
)

# --------------------------------
# EXPLANATION SECTION
# --------------------------------
with st.expander("📘 Feature Guide (What each field means)"):
    st.markdown("""
    ### Property Details Guide

    **Area**
    - Location of the property within Chennai.
    - Examples: Anna Nagar, Adyar, Velachery.

    **Interior Area (sq ft)**
    - Total built-up interior area in square feet.

    **Distance to Main Road**
    - Distance from property to nearest major road (in meters).

    **Bedrooms**
    - Number of bedrooms in the property.

    **Bathrooms**
    - Number of bathrooms.

    **Total Rooms**
    - Total count of all rooms.

    **Sale Condition**
    - Reason or type of sale:
        - **AdjLand** → Sale with adjacent land
        - **Partial** → Partially completed property
        - **Normal Sale** → Standard property sale
        - **AbNormal** → Urgent/distressed/non-standard sale
        - **Family** → Family-related transfer/sale

    **Parking Facility**
    - Whether parking space is available.

    **Building Type**
    - **House** → Residential house
    - **Commercial** → Commercial property
    - **Others** → Other building types

    **Utility Availability**
    Availability of essential utilities:
        - **AllPub** → All public utilities available
        - **NoSewa** → No sewage system
        - **NoSeWr** → No sewage + water restrictions
        - **ELO** → Electricity only

    **Street Type**
    Road accessibility:
        - **Paved** → Proper road
        - **Gravel** → Gravel road
        - **No Access** → Poor/no direct road access

    **Market Zone**
    Local zoning classification:
        - **RL** → Residential Low Density
        - **RH** → Residential High Density
        - **RM** → Residential Medium Density
        - **C** → Commercial
        - **A** → Agricultural / Mixed
        - **I** → Industrial

    **Room Quality Score**
    - Quality rating of rooms (0–10)

    **Bathroom Quality Score**
    - Bathroom construction/finish quality (0–10)

    **Bedroom Quality Score**
    - Bedroom quality rating (0–10)

    **Overall Quality Score**
    - Overall property condition rating (0–10)

    **Property Age**
    - Age of building in years.

    **Sale Year**
    - Year of sale.

    **Sale Month**
    - Month of sale.
    """)

st.divider()

# --------------------------------
# INPUT UI
# --------------------------------
col1, col2 = st.columns(2)

with col1:
    AREA = st.selectbox(
        "Area",
        [
            "Chrompet",
            "Karapakkam",
            "KK Nagar",
            "Velachery",
            "Anna Nagar",
            "Adyar",
            "T Nagar"
        ],
        help="Select the location of the property"
    )

    INT_SQFT = st.number_input(
        "Interior Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1200,
        help="Built-up area in square feet"
    )

    DIST_MAINROAD = st.number_input(
        "Distance to Main Road (meters)",
        min_value=0,
        max_value=10000,
        value=50,
        help="Distance from property to nearest main road"
    )

    N_BEDROOM = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    N_BATHROOM = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    N_ROOM = st.number_input(
        "Total Rooms",
        min_value=1,
        max_value=20,
        value=4
    )

    SALE_COND = st.selectbox(
        "Sale Condition",
        [
            "AdjLand",
            "Partial",
            "Normal Sale",
            "AbNormal",
            "Family"
        ]
    )

with col2:
    PARK_FACIL = st.selectbox(
        "Parking Facility",
        ["Yes", "No"]
    )

    BUILDTYPE = st.selectbox(
        "Building Type",
        [
            "House",
            "Others",
            "Commercial"
        ]
    )

    UTILITY_AVAIL = st.selectbox(
        "Utility Availability",
        [
            "AllPub",
            "NoSewa",
            "NoSeWr",
            "ELO"
        ]
    )

    STREET = st.selectbox(
        "Street Type",
        [
            "Paved",
            "Gravel",
            "No Access"
        ]
    )

    MZZONE = st.selectbox(
        "Market Zone",
        [
            "RL",
            "RH",
            "RM",
            "C",
            "A",
            "I"
        ]
    )

    QS_ROOMS = st.slider(
        "Room Quality Score",
        0.0, 10.0, 5.0
    )

    QS_BATHROOM = st.slider(
        "Bathroom Quality Score",
        0.0, 10.0, 5.0
    )

    QS_BEDROOM = st.slider(
        "Bedroom Quality Score",
        0.0, 10.0, 5.0
    )

    QS_OVERALL = st.slider(
        "Overall Quality Score",
        0.0, 10.0, 5.0
    )

st.divider()

col3, col4, col5 = st.columns(3)

with col3:
    PROPERTY_AGE = st.number_input(
        "Property Age (Years)",
        min_value=0,
        max_value=100,
        value=10
    )

with col4:
    SALE_YEAR = st.number_input(
        "Sale Year",
        min_value=2000,
        max_value=2035,
        value=2025
    )

with col5:
    SALE_MONTH = st.number_input(
        "Sale Month",
        min_value=1,
        max_value=12,
        value=5
    )

# --------------------------------
# PREDICT
# --------------------------------
if st.button("🔮 Predict Property Price", use_container_width=True):
    try:
        input_data = pd.DataFrame([{
            "AREA": AREA,
            "INT_SQFT": int(INT_SQFT),
            "DIST_MAINROAD": int(DIST_MAINROAD),
            "N_BEDROOM": int(N_BEDROOM),
            "N_BATHROOM": int(N_BATHROOM),
            "N_ROOM": int(N_ROOM),
            "SALE_COND": SALE_COND,
            "PARK_FACIL": PARK_FACIL,
            "BUILDTYPE": BUILDTYPE,
            "UTILITY_AVAIL": UTILITY_AVAIL,
            "STREET": STREET,
            "MZZONE": MZZONE,
            "QS_ROOMS": float(QS_ROOMS),
            "QS_BATHROOM": float(QS_BATHROOM),
            "QS_BEDROOM": float(QS_BEDROOM),
            "QS_OVERALL": float(QS_OVERALL),
            "PROPERTY_AGE": int(PROPERTY_AGE),
            "SALE_YEAR": int(SALE_YEAR),
            "SALE_MONTH": int(SALE_MONTH)
        }])

        prediction = model.predict(input_data)[0]
        predicted_price = np.expm1(prediction)

        st.success(f"💰 Estimated Property Price: ₹ {predicted_price:,.2f}")

        st.info(
            "Prediction is based on historical trends and should be treated as an estimate, not an official valuation."
        )

    except Exception as e:
        st.error(str(e))