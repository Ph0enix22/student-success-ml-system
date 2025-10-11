import streamlit as st
import pandas as pd
import pickle
import numpy as np

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# STYLING
st.markdown("""
    <style>
        .big-font {
            font-size: 2em !important;
            font-weight: 700;
        }
        .section-title {
            font-size: 1.15em;
            font-weight: 600;
            margin-top: 15px;
        }
        .key-factor {
            margin-bottom: 10px;
            font-size: 1.08em;
        }
        .risk {
            color: #b00020;
        }
        .positive {
            color: #008000;
        }
        .stButton>button {
            background: #2563eb;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# SIDEBAR - USER INSTRUCTIONS
with st.sidebar:
    st.header("📖 How To Use")
    st.write("1. Fill in student details")
    st.write("2. Click Predict")
    st.write("3. Review prediction, influencing factors & feature importance")
    st.markdown("---")
    st.markdown(
        "**Random Forest Classifier**  \n"
        "UCI Student Dataset  \n\n"
        "**Author:** SMJ  \n"
        "[GitHub](https://github.com/Ph0enix22)"
    )

# LOAD MODEL & ENCODERS
@st.cache_resource
def load_artifacts():
    """Load the trained Random Forest model and label encoders from disk."""
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run: `python src/train.py`")
        st.stop()

model, encoders = load_artifacts()

# MAIN TITLE & DESCRIPTION
st.markdown(
    '<div class="big-font">🎓 Student Performance Predictor</div>',
    unsafe_allow_html=True
)
st.caption("Machine Learning system for actionable student success insights.")
st.markdown("---")

# INPUT FORM - STUDENT DATA COLLECTION
with st.form("input_form"):
    st.markdown(
        '<div class="section-title">Student Information</div>',
        unsafe_allow_html=True
    )
    
    # Two-column layout for better organization
    col1, col2 = st.columns(2)
    
    # LEFT COLUMN - Demographics & Academic Background
    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Gender", ["M", "F"])
        age = st.slider("Age", 15, 22, 17)
        address = st.radio(
            "Address",
            ["U", "R"],
            help="U=Urban, R=Rural",
            horizontal=True
        )
        famsize = st.selectbox(
            "Family Size",
            ["LE3", "GT3"],
            help="LE3=≤3, GT3=>3"
        )
        Pstatus = st.selectbox(
            "Parent Status",
            ["T", "A"],
            help="T=Together, A=Apart"
        )
        Medu = st.select_slider(
            "Mother's Education",
            [0, 1, 2, 3, 4],
            2,
            format_func=lambda x: ["None", "Primary", "5-9th", "Secondary", "Higher"][x]
        )
        Fedu = st.select_slider(
            "Father's Education",
            [0, 1, 2, 3, 4],
            2,
            format_func=lambda x: ["None", "Primary", "5-9th", "Secondary", "Higher"][x]
        )
        Mjob = st.selectbox(
            "Mother's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )
        Fjob = st.selectbox(
            "Father's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )
        reason = st.selectbox(
            "School Choice Reason",
            ["home", "reputation", "course", "other"]
        )
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        traveltime = st.select_slider(
            "Travel Time",
            [1, 2, 3, 4],
            1,
            format_func=lambda x: ["<15min", "15-30min", "30-60min", ">60min"][x-1]
        )
        studytime = st.select_slider(
            "Study Time",
            [1, 2, 3, 4],
            2,
            format_func=lambda x: ["<2h", "2-5h", "5-10h", ">10h"][x-1]
        )
        failures = st.number_input("Past Failures", 0, 3, 0)
    
    # RIGHT COLUMN - Support, Activities & Social Factors
    with col2:
        schoolsup = st.selectbox("School Support", ["yes", "no"])
        famsup = st.selectbox("Family Support", ["yes", "no"])
        paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
        activities = st.selectbox("Extra Activities", ["yes", "no"])
        nursery = st.selectbox("Attended Nursery", ["yes", "no"])
        higher = st.selectbox("Wants Higher Ed", ["yes", "no"])
        internet = st.selectbox("Internet at Home", ["yes", "no"])
        romantic = st.selectbox("In Relationship", ["yes", "no"])
        famrel = st.slider("Family Relations", 1, 5, 4)
        freetime = st.slider("Free Time", 1, 5, 3)
        goout = st.slider("Going Out", 1, 5, 3)
        Dalc = st.slider("Workday Alcohol", 1, 5, 1)
        Walc = st.slider("Weekend Alcohol", 1, 5, 1)
        health = st.slider("Health Status", 1, 5, 3)
        absences = st.number_input("Absences", 0, 93, 0)
    
    submit = st.form_submit_button("🔮 Predict", use_container_width=True)

# PREDICTION & RESULTS DISPLAY
if submit:
    # Prepare input data in exact training order
    input_data = {
        'school': school, 'sex': sex, 'age': age, 'address': address,
        'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
        'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
        'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
        'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery, 'higher': higher,
        'internet': internet, 'romantic': romantic, 'famrel': famrel,
        'freetime': freetime, 'goout': goout, 'Dalc': Dalc, 'Walc': Walc,
        'health': health, 'absences': absences
    }
    input_df = pd.DataFrame([input_data])

    # Apply label encoding to categorical features
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Generate prediction and probability scores
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("---")
    
    # Display prediction result with probability
    if prediction == 1:
        st.success("🎉 Predicted: PASS")
        st.metric("PASS Probability", f"{probability[1]*100:.1f}%")
        st.info("This student is likely to succeed. Encourage continued effort and positive habits!")
    else:
        st.error("🚩 Predicted: FAIL")
        st.metric("FAIL Probability", f"{probability[0]*100:.1f}%")
        st.warning("Recommend additional academic support and intervention.")

    st.markdown("---")
    
    # Display feature importance from Random Forest
    if hasattr(model, 'feature_importances_'):
        st.markdown(
            '<div class="section-title">📊 Top Features Influencing Prediction</div>',
            unsafe_allow_html=True
        )
        importances = pd.DataFrame({
            'Feature': input_df.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(5)
        st.bar_chart(importances.set_index('Feature'))
    
    st.markdown("---")

    # Display key performance factors (based on EDA insights)
    st.markdown(
        '<div class="section-title">🔍 Key Performance Factors</div>',
        unsafe_allow_html=True
    )
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("**⚠️ Risk Indicators:**")
        if failures > 0:
            st.markdown(
                f'<div class="key-factor risk">• {failures} past failures</div>',
                unsafe_allow_html=True
            )
        if absences > 10:
            st.markdown(
                f'<div class="key-factor risk">• High absences: {absences}</div>',
                unsafe_allow_html=True
            )
        if studytime < 2:
            st.markdown(
                '<div class="key-factor risk">• Limited study time</div>',
                unsafe_allow_html=True
            )
        if higher == "no":
            st.markdown(
                '<div class="key-factor risk">• No higher education plans</div>',
                unsafe_allow_html=True
            )
    
    with colB:
        st.markdown("**✅ Positive Indicators:**")
        if failures == 0:
            st.markdown(
                '<div class="key-factor positive">• No past failures</div>',
                unsafe_allow_html=True
            )
        if absences < 5:
            st.markdown(
                '<div class="key-factor positive">• Excellent attendance</div>',
                unsafe_allow_html=True
            )
        if studytime >= 3:
            st.markdown(
                '<div class="key-factor positive">• Strong study habits</div>',
                unsafe_allow_html=True
            )
        if higher == "yes":
            st.markdown(
                '<div class="key-factor positive">• Aspires for higher education</div>',
                unsafe_allow_html=True
            )

# FOOTER
st.markdown("---")
st.caption(
    "Built by SMJ • Student Performance ML System\n"
    "**Model:** RandomForestClassifier + LabelEncoder, 80/20 train/test, "
    "Tuned with GridSearchCV.  \n"
    "**Metrics:** F1≈0.78, Acc≈0.66 (Tuned Decision Tree)  \n\n"
    "Made with ❤️"
)