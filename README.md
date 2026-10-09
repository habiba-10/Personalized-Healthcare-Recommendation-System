# 🏥 Personalized Healthcare Recommendation System

An end-to-end Machine Learning healthcare platform that routes patients to the correct clinical specialty based on 131 active symptoms and provides a multi-criteria provider ranking engine. Built with Scikit-Learn, CatBoost, XGBoost, and Streamlit.

---

## 📸 Overview & Key Features

- **Multi-Class Clinical Routing (Stage 1)**: Classifies reported patient symptoms into required medical specialties using Linear Discriminant Analysis (LDA) and Random Forest / CatBoost / XGBoost ensemble models.
- **Two-Stage Provider Scoring Engine (Stage 2)**: Dynamically ranks hospitals and physicians using a composite weighted quality score combining CMS Hospital Ratings (40%), Patient Satisfaction (30%), and Clinical Experience (30%).
- **Interactive Streamlit Web Dashboard**: Live user portal for selecting patient records, checking historical booking activity, and receiving immediate provider recommendations.
- **Dimensionality Reduction**: Utilizes LDA to compress high-dimensional symptom matrices while preserving max class separation between medical specialties.

---

## 🛠️ Tech Stack & Dependencies

- **Frontend / Application Framework**: Streamlit
- **Machine Learning & Pipeline**: Scikit-Learn, XGBoost, CatBoost, Joblib
- **Dimensionality Reduction**: Linear Discriminant Analysis (LDA)
- **Data Manipulation & Analysis**: Pandas, NumPy, OpenPyXL
- **Visualization & EDA**: Matplotlib, Seaborn

---

## 📁 Repository Structure

Personalized_Healthcare_Recommendation/
├── app.py                            # Interactive Streamlit web application
├── data_enrichment.ipynb             # ML Pipeline (EDA, Feature Engineering, LDA, Training)
├── datasets/                         # Medical & CMS datasets
│   ├── appointments.csv
│   ├── doctors.csv
│   ├── patients.csv
│   ├── Doctor_Specialist.csv
│   ├── Doctor_Versus_Disease.csv
│   ├── Disease_Description.csv
│   ├── Original_Dataset.csv
│   ├── Symptom_weights.csv
│   └── Hospital_General_Information_2016_2020.xlsx
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Git exclusion rules

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
git clone https://github.com/habiba-10/Personalized-Healthcare-Recommendation-System.git
cd Personalized-Healthcare-Recommendation-System

### 2. Set Up Virtual Environment & Install Dependencies
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Run the Streamlit Application
streamlit run app.py

---

## ⚙️ Model Architecture & Recommendation Pipeline

1. **Relational Data Fusion**: Merged patient profiles, booking history, doctor credentials, symptom-disease mappings, and CMS quality ratings.
2. **Symptom Vectorization**: Flattened unstructured symptom records into a binary matrix covering 131 clinical symptoms.
3. **Dimensionality Reduction**: Applied LDA to extract optimal discriminant features and simplify decision boundaries.
4. **Classification Models**: Trained Random Forest, CatBoost, and XGBoost classifiers for diagnostic specialty prediction.
5. **Quality-Weighted Ranking**: Filtered matching providers and evaluated them using a multi-criteria scoring function.

---

## 👤 Author
**Habiba Ezat**
- GitHub: [@habiba-10](https://github.com/habiba-10)

**Habiba Ezat**
- GitHub: [@habiba-10](https://github.com/habiba-10)
