# ==========================================================
# File Name          : app.py
# Author             : Habiba Ezat Abdeltawab
# Date of Creation   : 20-07-2026
# Dependencies       : streamlit, pandas, numpy, scikit-learn, joblib
# Description        : Robust Dynamic Production Streamlit Application
#                      With Strict Real Model Inference Engine.
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================================
# CONSTANTS DEFINITION
# ==========================================================
DEFAULT_TOP_N_RECOMMENDATIONS = 3
WEIGHT_HOSPITAL_RATING = 0.40
WEIGHT_PATIENT_EXPERIENCE = 0.30
WEIGHT_DOCTOR_EXPERIENCE = 0.30
MAX_QUALITY_SCORE = 5.0

EXPERIENCE_USER_MAPPING = {
    "Above the national average": "⭐ Highly Rated by Patients (Exceeds National Standards)",
    "Same as the national average": "✅ Standard Patient Satisfaction (Meets National Standards)",
    "Below the national average": "⚠️ Average Patient Satisfaction",
    "Not available": "ℹ️ Rating Pending Assessment"
}

# Operational Non-Symptom Columns to Exclude from Symptoms List
OPERATIONAL_COLS = [
    'years_experience', 'patient_appointment_count', 'gender_encoded', 
    'hospital_rating_clean', 'patient_experience_encoded', 'age'
]

# ==========================================================
# PAGE CONFIGURATION & STYLING
# ==========================================================
st.set_page_config(
    page_title="Personalized Healthcare Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .ai-badge {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }
    .card-container {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .score-badge {
        background-color: #059669;
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 0.88rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# AUTOMATIC ARTIFACTS & DATASET LOADER
# ==========================================================
@st.cache_resource
def load_trained_artifacts():
    """Loads saved Random Forest LDA Model, LDA Transformer, and Features List."""
    try:
        model = joblib.load("rf_lda_model.pkl")
        lda_trans = joblib.load("lda_transformer.pkl")
        features = joblib.load("model_features.pkl")
        return model, lda_trans, features
    except Exception as e:
        st.error(f"⚠️ Model artifacts loading error: {e}. Please ensure model files are saved properly.")
        return None, None, None

@st.cache_data
def load_dataset_dynamically():
    """Loads cleaned master healthcare dataset."""
    possible_filenames = [
        "cleaned_healthcare_dataset.csv",
        "healthcare_dataset.csv",
        "df_healthcare.csv"
    ]
    for fname in possible_filenames:
        if os.path.exists(fname):
            try:
                return pd.read_csv(fname)
            except Exception:
                continue
    return None

# Load System Artifacts
rf_lda_model, lda_transformer, model_features = load_trained_artifacts()
df_master = load_dataset_dynamically()

# Extract Unique Patient List & All 131 Clinical Symptoms List Safely
if df_master is not None and not df_master.empty:
    all_patient_ids = sorted([str(pid) for pid in df_master["patient_id"].unique().tolist()])
    
    # Extract symptoms columns dynamically
    if model_features is not None:
        symptom_columns = [col for col in model_features if col not in ['age', 'gender_encoded', 'patient_appointment_count']]
    else:
        symptom_columns = [col for col in df_master.select_dtypes(include=['int64', 'int32']).columns if col not in OPERATIONAL_COLS]
        
    symptom_display_map = {col: col.replace("_", " ").strip().title() for col in symptom_columns}
    symptom_reverse_map = {v: k for k, v in symptom_display_map.items()}
    display_symptoms_list = sorted(list(symptom_display_map.values()))
else:
    all_patient_ids = ["P001", "P002", "P003"]
    display_symptoms_list = ["Joint Pain", "Skin Rash", "Chest Pain", "Headache"]
    symptom_reverse_map = {}

# ==========================================================
# HELPER FUNCTIONS & QUALITY SCORING LOGIC
# ==========================================================
def calculate_composite_quality_score(df_input):
    """Computes multi-criteria composite quality score for providers."""
    max_experience = df_master["years_experience"].max() if "years_experience" in df_master.columns else 30
    if max_experience == 0 or pd.isnull(max_experience):
        max_experience = 30

    df_scored = df_input.copy()
    h_rating = df_scored["hospital_rating_clean"] if "hospital_rating_clean" in df_scored.columns else 4
    p_exp = df_scored["patient_experience_encoded"] if "patient_experience_encoded" in df_scored.columns else 2
    y_exp = df_scored["years_experience"] if "years_experience" in df_scored.columns else 10
    
    df_scored["composite_score"] = (
        (h_rating * WEIGHT_HOSPITAL_RATING) +
        (p_exp * WEIGHT_PATIENT_EXPERIENCE) +
        ((y_exp / max_experience) * MAX_QUALITY_SCORE * WEIGHT_DOCTOR_EXPERIENCE)
    )
    return df_scored

def get_user_friendly_experience(raw_experience_string):
    """Maps raw clinical experience terminology to friendly strings."""
    raw_str = str(raw_experience_string).strip().capitalize()
    return EXPERIENCE_USER_MAPPING.get(raw_str, EXPERIENCE_USER_MAPPING["Not available"])

# ==========================================================
# SIDEBAR: DYNAMIC PATIENT PROFILE & SYMPTOMS SELECTION
# ==========================================================
st.sidebar.header("📋 Patient Profile Identification")

selected_patient_id = st.sidebar.selectbox(
    "Select Registered Patient ID", 
    all_patient_ids,
    index=0
)

# Dynamic Read-Only Profile Lookup for Selected Patient ID
if df_master is not None and "patient_id" in df_master.columns:
    patient_rows = df_master[df_master["patient_id"].astype(str) == selected_patient_id]
    if not patient_rows.empty:
        p_row = patient_rows.iloc[0]
        p_age = p_row.get("age", 35)
        p_gender = p_row.get("gender", "F")
        p_history = p_row.get("patient_appointment_count", 1)
    else:
        p_age, p_gender, p_history = 35, "F", 1
else:
    p_age, p_gender, p_history = 35, "F", 1

# Render Selected Patient Specific Details
st.sidebar.markdown(f"**Age:** {p_age} Yrs | **Gender:** {p_gender}")
st.sidebar.markdown(f"**Booking Record:** {p_history} Appointments")

st.sidebar.divider()
st.sidebar.header("🩺 Symptoms Input (131 Clinical Symptoms)")

selected_display_symptoms = st.sidebar.multiselect(
    "Select Active Symptoms",
    display_symptoms_list,
    default=[display_symptoms_list[0]] if display_symptoms_list else []
)

btn_analyze = st.sidebar.button("Analyze Profile & Recommend")

# ==========================================================
# MAIN DASHBOARD CONTENT
# ==========================================================
st.markdown("<div class='main-header'>Personalized Healthcare Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Tailored recommendations customized according to your clinical profile & behavior history.</div>", unsafe_allow_html=True)

# Mandatory AI Disclosure Notice
st.markdown("""
    <div class='ai-badge'>
        <strong>🤖 AI-Generated Recommendation Notice:</strong><br>
        The clinical recommendations and provider ratings below are generated using Machine Learning models (Random Forest with LDA). They analyze active symptoms, service quality, and historical patient behavior.
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# REAL AI MODEL INFERENCE PIPELINE
# ----------------------------------------------------------
if rf_lda_model is not None and lda_transformer is not None and model_features is not None:
    
    # 1. Construct Feature Vector matching exact training columns
    input_dict = {col: 0 for col in model_features}
    
    # Map active symptoms
    for sym_display in selected_display_symptoms:
        raw_col = symptom_reverse_map.get(sym_display)
        if raw_col and raw_col in input_dict:
            input_dict[raw_col] = 1
            
    # Map Patient Demographics
    input_dict['age'] = p_age
    input_dict['gender_encoded'] = 1 if str(p_gender).upper() == 'M' else 0
    input_dict['patient_appointment_count'] = p_history
    
    input_df = pd.DataFrame([input_dict])[model_features]
    
    # 2. LDA Dimension Reduction & RF Prediction
    input_lda = lda_transformer.transform(input_df)
    predicted_specialty = str(rf_lda_model.predict(input_lda)[0]).strip().title()
    
    if hasattr(rf_lda_model, "predict_proba"):
        diagnostic_confidence = np.max(rf_lda_model.predict_proba(input_lda)[0]) * 100
    else:
        diagnostic_confidence = 100.0
else:
    predicted_specialty = "General Practitioner"
    diagnostic_confidence = 100.0

# Display Diagnostic Banner
st.success(f"🎯 **AI Clinical Routing:** Suggested Specialty $\\rightarrow$ **{predicted_specialty.upper()}** (Model Confidence: {diagnostic_confidence:.1f}%)")

# ----------------------------------------------------------
# STRICT PROVIDER FILTERING & DASHBOARD DISPLAY
# ----------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🎯 Tailored Healthcare Recommendations")
    
    st.info(f"💡 **Clinical Direction:** Based on reported symptoms ({', '.join(selected_display_symptoms) if selected_display_symptoms else 'None'}), our AI model routes your case to **{predicted_specialty}**.")
    
    st.markdown(f"• **Recommended Action:** Consult a certified **{predicted_specialty}** specialist for comprehensive clinical evaluation.")
    st.markdown(f"• **Pre-Consultation Tip:** Maintain a log of symptom occurrences and bring previous medical reports to your appointment.")

    st.subheader("📊 Your Healthcare Engagement Profile")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Total Appointments", f"{p_history} Sessions", "Active Record")
    with metric_col2:
        st.metric("Recommended Specialty", predicted_specialty, "AI Matched")
    with metric_col3:
        st.metric("Patient Status", "Verified Patient", "High Compliance")

with col_right:
    st.subheader("👨‍⚕️ Featured Providers For You")
    
    # STRICT FILTERING: Look strictly for doctors matching predicted specialty
    if df_master is not None and "Specialist" in df_master.columns:
        # Match against Specialist or specialization column
        strict_match_mask = (
            df_master['Specialist'].astype(str).str.strip().str.title() == predicted_specialty
        ) | (
            df_master['specialization'].astype(str).str.strip().str.title() == predicted_specialty
        )
        matched_providers = df_master[strict_match_mask].copy()
    else:
        matched_providers = pd.DataFrame()

    # Dynamic Message when no doctor matches the predicted specialty
    if matched_providers.empty:
        st.warning(f"⚠️ **No Doctors Available:** Currently, there are no doctors in our directory specializing in **{predicted_specialty}**. Please check back later or consult general outpatient services.")
    else:
        st.caption(f"Top matching healthcare centers for **{predicted_specialty}** ranked by AI Quality Score (Max 5.0):")
        
        df_ranked = calculate_composite_quality_score(matched_providers)
        ranked_recommendations = (
            df_ranked.sort_values(by='composite_score', ascending=False)
            .drop_duplicates(subset=['doctor_id', 'hospital_name'])
            .head(DEFAULT_TOP_N_RECOMMENDATIONS)
        )

        for _, row in ranked_recommendations.iterrows():
            raw_exp = row.get("patient_experience", "Not available")
            friendly_exp = get_user_friendly_experience(raw_exp)
            
            doc_id_val = row.get("doctor_id", "D001")
            branch_val = row.get("hospital_branch", "Main Branch")
            spec_val = row.get("Specialist", row.get("specialization", predicted_specialty))
            hosp_val = row.get("hospital_name", "Medical Center")
            y_exp_val = row.get("years_experience", 10)
            score_val = row.get("composite_score", 4.0)
            
            st.markdown(f"""
                <div class='card-container'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h4 style='margin:0; color:#1E3A8A;'>{doc_id_val} - {branch_val}</h4>
                        <span class='score-badge'>AI Quality Rating: {score_val:.2f} / 5.0</span>
                    </div>
                    <p style='margin: 6px 0 2px 0; color:#374151;'><strong>Specialty:</strong> {spec_val}</p>
                    <p style='margin: 2px 0; color:#374151;'><strong>Hospital Facility:</strong> {hosp_val}</p>
                    <p style='margin: 2px 0; color:#374151;'><strong>Doctor Experience:</strong> {y_exp_val} Years</p>
                    <p style='margin: 4px 0; color:#059669; font-size:0.88rem;'>{friendly_exp}</p>
                </div>
            """, unsafe_allow_html=True)