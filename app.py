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
    st.write("3. Review prediction & recommendations")
    st.markdown("---")
    st.markdown(
        "**Decision Tree Classifier (Tuned)**  \n"
        "UCI Student Dataset  \n\n"
        "**Author:** SMJ  \n"
        "[GitHub](https://github.com/Ph0enix22)"
    )

# LOAD MODEL & ENCODERS
@st.cache_resource
def load_artifacts():
    """Load the trained Decision Tree model and label encoders from disk."""
    try:
        with open('models/decision_tree_model.pkl', 'rb') as f:
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
st.caption("ML system to predict if a student will pass or need support.")
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
        school_display = st.selectbox("School Type", ["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"])
        school = "GP" if "Gabriel" in school_display else "MS"
        
        sex_display = st.selectbox("Gender", ["Male", "Female"])
        sex = "M" if sex_display == "Male" else "F"
        
        age = st.slider("Age", 15, 22, 17)
        
        address_display = st.radio(
            "Address Type",
            ["Urban", "Rural"],
            horizontal=True
        )
        address = "U" if address_display == "Urban" else "R"
        
        famsize_display = st.selectbox(
            "Family Size",
            ["Small (≤3 members)", "Large (>3 members)"]
        )
        famsize = "LE3" if "Small" in famsize_display else "GT3"
        
        pstatus_display = st.selectbox(
            "Parent Status",
            ["Living Together", "Apart"]
        )
        Pstatus = "T" if "Together" in pstatus_display else "A"
        
        Medu = st.select_slider(
            "Mother's Education Level",
            [0, 1, 2, 3, 4],
            2,
            format_func=lambda x: ["No education", "Primary school", "5-9th grade", "Secondary school", "Higher education"][x]
        )
        
        Fedu = st.select_slider(
            "Father's Education Level",
            [0, 1, 2, 3, 4],
            2,
            format_func=lambda x: ["No education", "Primary school", "5-9th grade", "Secondary school", "Higher education"][x]
        )
        
        Mjob = st.selectbox(
            "Mother's Job",
            ["Teacher", "Healthcare", "Services", "At Home", "Other"]
        )
        Mjob_map = {"Teacher": "teacher", "Healthcare": "health", "Services": "services", "At Home": "at_home", "Other": "other"}
        Mjob = Mjob_map[Mjob]
        
        Fjob = st.selectbox(
            "Father's Job",
            ["Teacher", "Healthcare", "Services", "At Home", "Other"]
        )
        Fjob = Mjob_map[Fjob]
        
        reason_display = st.selectbox(
            "Reason for School Choice",
            ["Close to Home", "School Reputation", "Course Preference", "Other"]
        )
        reason_map = {"Close to Home": "home", "School Reputation": "reputation", "Course Preference": "course", "Other": "other"}
        reason = reason_map[reason_display]
        
        guardian = st.selectbox("Primary Guardian", ["Mother", "Father", "Other"])
        guardian = guardian.lower()
        
        traveltime = st.select_slider(
            "Travel Time to School",
            [1, 2, 3, 4],
            1,
            format_func=lambda x: ["<15 minutes", "15-30 minutes", "30-60 minutes", ">60 minutes"][x-1]
        )
        
        studytime = st.select_slider(
            "Weekly Study Time",
            [1, 2, 3, 4],
            2,
            format_func=lambda x: ["<2 hours", "2-5 hours", "5-10 hours", ">10 hours"][x-1]
        )
        
        failures = st.number_input("Number of Past Class Failures", 0, 3, 0)
    
    # RIGHT COLUMN - Support, Activities & Social Factors
    with col2:
        schoolsup_display = st.selectbox("School Support Services", ["Yes", "No"])
        schoolsup = schoolsup_display.lower()
        
        famsup_display = st.selectbox("Family Educational Support", ["Yes", "No"])
        famsup = famsup_display.lower()
        
        paid_display = st.selectbox("Takes Extra Paid Classes", ["Yes", "No"])
        paid = paid_display.lower()
        
        activities_display = st.selectbox("Participates in Extra Activities", ["Yes", "No"])
        activities = activities_display.lower()
        
        nursery_display = st.selectbox("Attended Nursery School", ["Yes", "No"])
        nursery = nursery_display.lower()
        
        higher_display = st.selectbox("Wants to Pursue Higher Education", ["Yes", "No"])
        higher = higher_display.lower()
        
        internet_display = st.selectbox("Has Internet at Home", ["Yes", "No"])
        internet = internet_display.lower()
        
        romantic_display = st.selectbox("In a Romantic Relationship", ["Yes", "No"])
        romantic = romantic_display.lower()
        
        famrel = st.slider("Quality of Family Relationships", 1, 5, 4, 
                          help="1 = Very Poor, 5 = Excellent")
        
        freetime = st.slider("Amount of Free Time After School", 1, 5, 3,
                            help="1 = Very Low, 5 = Very High")
        
        goout = st.slider("Frequency of Going Out with Friends", 1, 5, 3,
                         help="1 = Very Low, 5 = Very High")
        
        Dalc = st.slider("Weekday Alcohol Consumption", 1, 5, 1,
                        help="1 = Very Low/None, 5 = Very High")
        
        Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1,
                        help="1 = Very Low/None, 5 = Very High")
        
        health = st.slider("Current Health Status", 1, 5, 3,
                          help="1 = Very Bad, 5 = Very Good")
        
        absences = st.number_input("Number of School Absences (this period)", 0, 93, 0)
    
    submit = st.form_submit_button("🔮 Predict Performance", use_container_width=True)

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
        st.metric("Pass Probability", f"{probability[1]*100:.1f}%")
        st.info("This student is likely to succeed. Encourage continued effort and positive habits!")
    else:
        st.error("🚩 Predicted: FAIL")
        st.metric("Fail Probability", f"{probability[0]*100:.1f}%")
        st.warning("This student may need additional academic support and intervention.")

    st.markdown("---")
    
    # Display feature importance from Random Forest
    if hasattr(model, 'feature_importances_'):
        st.markdown(
            '<div class="section-title">📊 General Feature Importance of the model for All Students</div>',
            unsafe_allow_html=True
        )
        importances = pd.DataFrame({
            'Factor': input_df.columns,
            'Influence': model.feature_importances_
        }).sort_values('Influence', ascending=False).head(5)
        st.bar_chart(importances.set_index('Factor'))
    
    st.markdown("---")

    # Display key performance factors (based on EDA insights)
    st.markdown(
        '<div class="section-title">🔍 Key Performance Indicators</div>',
        unsafe_allow_html=True
    )
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("**⚠️ Areas of Concern:**")
        if failures > 0:
            st.markdown(
                f'<div class="key-factor risk">• Previous failures: {failures}</div>',
                unsafe_allow_html=True
            )
        if absences > 10:
            st.markdown(
                f'<div class="key-factor risk">• High absences: {absences} days</div>',
                unsafe_allow_html=True
            )
        if studytime < 2:
            st.markdown(
                '<div class="key-factor risk">• Limited study time (<5 hours/week)</div>',
                unsafe_allow_html=True
            )
        if higher_display == "No":
            st.markdown(
                '<div class="key-factor risk">• No higher education plans</div>',
                unsafe_allow_html=True
            )
        if not any([failures > 0, absences > 10, studytime < 2, higher_display == "No"]):
            st.markdown('<div class="key-factor">✓ No major concerns identified</div>', unsafe_allow_html=True)
    
    with colB:
        st.markdown("**✅ Positive Factors:**")
        if failures == 0:
            st.markdown(
                '<div class="key-factor positive">• No previous failures</div>',
                unsafe_allow_html=True
            )
        if absences < 5:
            st.markdown(
                '<div class="key-factor positive">• Excellent attendance</div>',
                unsafe_allow_html=True
            )
        if studytime >= 3:
            st.markdown(
                '<div class="key-factor positive">• Strong study habits (>5 hours/week)</div>',
                unsafe_allow_html=True
            )
        if higher_display == "Yes":
            st.markdown(
                '<div class="key-factor positive">• Motivated for higher education</div>',
                unsafe_allow_html=True
            )
        if famsup_display == "Yes":
            st.markdown(
                '<div class="key-factor positive">• Strong family support</div>',
                unsafe_allow_html=True
            )

# FOOTER
st.markdown("---")
st.caption(
    "Built by SMJ • Student Performance ML System\n"
    "**Model:** Decision Tree (Tuned), 80/20 train/test, GridSearchCV  \n"
    "**Performance:** F1≈0.78, Accuracy≈0.66 on 395 students\n\n"
    "Made with ❤️"
)
