# 🏥 Personalized Healthcare Recommendation System

An end-to-end, two-stage ML system that (1) **routes a patient to the right medical specialty** from 131 binary symptoms, then (2) **ranks doctors and hospitals** with a weighted quality score. It is served through a Streamlit dashboard.

## 🧩 How It Works

| Stage | Task | Method |
|---|---|---|
| **Stage 1** | Clinical routing: predict the required specialist (16 classes) | LDA dimensionality reduction + Random Forest / CatBoost / XGBoost |
| **Stage 2** | Provider ranking: recommend top-N doctors in the predicted specialty | Composite score: **40%** CMS hospital rating + **30%** patient satisfaction + **30%** clinical experience |

**Example output** (patient `P012`, top 3):

```
STAGE 1: Predicted specialist: OTOLARYNGOLOGIST (model confidence 51%)
STAGE 2: Top recommended doctors & clinics
  #1  D009 (Central Hospital)  | 26 yrs | CMS 4/5 | Above national average | Service index 3.89/5
  #2  D005 (Central Hospital)  | 26 yrs | CMS 4/5 | Above national average | Service index 3.89/5
  #3  D006 (Central Hospital)  | 23 yrs | CMS 4/5 | Above national average | Service index 3.73/5
```

## 📊 Stage 1 Results (40-sample test set, 16 specialties)

| Model | Before LDA (Acc.) | After LDA (Acc.) | After LDA (Macro F1) |
|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 1.000 |
| CatBoost | 0.975 | **1.000** | 1.000 |
| XGBoost | 0.950 | 0.950 | 0.974 |

- LDA compressed **134 input features (131 symptoms + age + gender + appointment history) into 15 discriminant components** (the maximum allowed for 16 classes) while preserving class separation.
- LDA lifted CatBoost from 97.5% to 100% test accuracy and improved XGBoost's macro F1 from 0.954 to 0.974.

> **How to read these numbers:** the dataset is small (200 appointment records, 50 patients) and the test set has only 40 samples, so near-perfect scores mainly show that symptom patterns map cleanly to specialties in this data. They are **not** evidence of real-world clinical accuracy. See Limitations.

## 🗂️ Data

Relational data fused into one master table (200 rows × 155 columns): patients, appointments, doctors, doctor-specialty and disease-specialty mappings, symptom weights, and CMS hospital quality ratings (2016–2020).

Preprocessing highlights: age derived from date of birth with logical-error checks, median imputation for missing hospital ratings, ordinal encoding of patient satisfaction, gender encoding, a patient appointment-count feature, and a 131-column binary symptom matrix.

## 🛠️ Tech Stack

Streamlit · Scikit-Learn · XGBoost · CatBoost · Joblib · Pandas · NumPy · OpenPyXL · Matplotlib · Seaborn

## 📁 Repository Structure

```
Personalized-Healthcare-Recommendation-System/
├── app.py                    # Interactive Streamlit web application
├── data_enrichment.ipynb     # Full ML pipeline: EDA, feature engineering, LDA, training
├── datasets/                 # Medical, appointment, and CMS datasets
│   ├── appointments.csv
│   ├── doctors.csv
│   ├── patients.csv
│   ├── Doctor_Specialist.csv
│   ├── Doctor_Versus_Disease.csv
│   ├── Disease_Description.csv
│   ├── Original_Dataset.csv
│   ├── Symptom_weights.csv
│   └── Hospital_General_Information_2016_2020.xlsx
├── requirements.txt
└── .gitignore
```

## 🚀 Quickstart

```bash
git clone https://github.com/habiba-10/Personalized-Healthcare-Recommendation-System.git
cd Personalized-Healthcare-Recommendation-System
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Limitations

- **Small data:** 200 records and a 40-sample test set make the metrics unstable. A single misclassified sample moves accuracy by 2.5 points.
- **Partly synthetic fields:** some values (for example, pediatric patient ages) were generated to keep the data logically consistent, so results reflect this dataset rather than real patients.
- **Modest confidence on real inputs:** the sample patient above was routed with 51% confidence, so low-confidence predictions should trigger a human review rather than an automatic decision.
- Educational prototype, **not a medical device** and not a substitute for professional diagnosis.

## 🔭 Future Work

- Validate with stratified k-fold cross-validation instead of a single small split.
- Surface prediction confidence and the top-3 specialties in the app.
- Evaluate on a larger, real-world symptom dataset.
- Add personalization from booking history to the Stage 2 ranking.

## 👤 Author

**Habiba Ezat** · [GitHub](https://github.com/habiba-10) · [LinkedIn](https://www.linkedin.com/in/habiba-ezat-a39521293)
