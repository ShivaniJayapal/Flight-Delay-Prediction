# ✈️ Flight Delay Prediction — Advanced ML Pipeline

> Predicting US domestic flight delays using machine learning on 2024 Bureau of Transportation Statistics data.  
> **Best Model: Logistic Regression — ROC-AUC: 0.9998 | F1-Score: 0.9902**

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Features Engineered](#-features-engineered)
- [ML Pipeline](#-ml-pipeline)
- [Model Results](#-model-results)
- [EDA Highlights](#-eda-highlights)
- [How to Run](#-how-to-run)
- [Single Flight Prediction](#-single-flight-prediction)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)

---

## 🔍 Project Overview

This project builds an end-to-end machine learning pipeline to predict whether a US domestic flight will be **delayed by more than 15 minutes**. It covers the full data science workflow:

- ✅ Data cleaning & preprocessing
- ✅ Advanced feature engineering (18+ new features)
- ✅ Exploratory Data Analysis with 11 visualizations
- ✅ Class imbalance handling with **SMOTE**
- ✅ 5 models trained + **Voting Ensemble**
- ✅ Full evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR Curves
- ✅ **SHAP explainability** — understand why a flight is predicted delayed
- ✅ Route risk analysis — identify the most delay-prone routes
- ✅ Model saving & a **custom single-flight predictor**

---

## 📦 Dataset

| Property | Value |
|---|---|
| Source | US Bureau of Transportation Statistics (BTS) |
| Year | 2024 |
| Rows | ~10,000 (sample) |
| Columns | 35 raw → 52 after engineering |
| Target | `Is_Delayed` — arrival delay > 15 minutes |
| Overall Delay Rate | **20.4%** |

### Key Raw Columns Used

| Column | Description |
|---|---|
| `op_unique_carrier` | Airline carrier code (WN, AA, DL, etc.) |
| `origin` / `dest` | Origin and destination airport codes |
| `crs_dep_time` | Scheduled departure time (HHMM format) |
| `arr_delay` | Arrival delay in minutes (target source) |
| `distance` | Route distance in miles |
| `crs_elapsed_time` | Scheduled flight duration |
| `carrier_delay` | Minutes delayed due to carrier |
| `weather_delay` | Minutes delayed due to weather |
| `nas_delay` | Minutes delayed due to National Air System |
| `security_delay` | Minutes delayed due to security |
| `late_aircraft_delay` | Minutes delayed due to late arriving aircraft |
| `day_of_week` | Day of week (1=Mon … 7=Sun) |
| `month` | Month (1–12) |

---

## Project Structure

```
flight-delay-prediction/
│
├── flight_delay_prediction_final.py   # Full pipeline (single script)
│
├── notebooks/
│   └── Flight_Delay_Colab.ipynb       # 23-cell Google Colab notebook
│
├── plots/                             # All generated EDA & evaluation plots
│   ├── 01_delay_overview.png
│   ├── 02_delay_by_hour_day.png
│   ├── 03_delay_by_month_season_distance.png
│   ├── 04_airline_analysis.png
│   ├── 05_route_risk.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_confusion_matrices.png
│   ├── 08_roc_pr_curves.png
│   ├── 09_feature_importance.png
│   ├── 10_shap_summary.png
│   └── 11_shap_bar.png
│
├── model_artifacts/                   # Saved models & preprocessors
│   ├── best_model.pkl
│   ├── all_models.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── feature_cols.pkl
│
├── data/
│   └── flight_data_2024_sample.csv    # Raw dataset (place here)
│
├── requirements.txt
└── README.md
```

---

## Features Engineered

Starting from 35 raw columns, **18+ new features** were created:

### Time Features
| Feature | Description |
|---|---|
| `dep_hour` | Departure hour extracted from `crs_dep_time` |
| `dep_minute` | Departure minute |
| `dep_time_bin` | 6 time-of-day bins (Red-Eye / Early / Morning / Afternoon / Evening / Night) |
| `peak_hour` | 1 if morning rush (6–9am) or evening rush (5–9pm) |
| `red_eye` | 1 if flight departs midnight–5am or 10pm+ |
| `weekend` | 1 if Saturday or Sunday |
| `is_monday` | 1 if Monday (high disruption day) |
| `is_friday` | 1 if Friday (high disruption day) |
| `season` | 1=Winter 2=Spring 3=Summer 4=Fall |
| `quarter` | Calendar quarter (1–4) |

### Route & Distance Features
| Feature | Description |
|---|---|
| `distance_bin` | Short (<500mi) / Medium / Long / Ultra (>1500mi) |
| `route` | `ORIGIN-DEST` string for grouping |

### 📊 Historical Delay Rate Features (Target Encoding)
| Feature | Description |
|---|---|
| `airline_delay_rate` | Historical delay % for this carrier |
| `origin_delay_rate` | Historical delay % at origin airport |
| `dest_delay_rate` | Historical delay % at destination airport |
| `route_delay_rate` | Historical delay % for this exact route |
| `fl_num_freq` | How often this flight number appears (busy route proxy) |

---

## 🔧 ML Pipeline

```
Raw CSV
  │
  ├─ 1. Data Cleaning
  │     • Fill NaN delay cause cols with 0
  │     • Drop columns with >60% missing
  │     • Median/mode imputation
  │     • Remove duplicates
  │
  ├─ 2. Feature Engineering
  │     • 18+ new time, route & historical features
  │
  ├─ 3. Outlier Handling
  │     • Winsorization (IQR method) on delay & distance cols
  │
  ├─ 4. Label Encoding
  │     • airline, origin, dest → integer codes
  │
  ├─ 5. Train/Test Split  (80% / 20%, stratified)
  │
  ├─ 6. SMOTE  (oversample minority class to fix imbalance)
  │
  ├─ 7. StandardScaler
  │
  ├─ 8. Model Training (5-Fold CV on each)
  │     • Random Forest
  │     • XGBoost
  │     • Gradient Boosting
  │     • Logistic Regression
  │     • Voting Ensemble (RF + XGB + GB, soft voting)
  │
  └─ 9. Evaluation
        • Accuracy, Precision, Recall, F1, ROC-AUC
        • Confusion Matrices
        • ROC & Precision-Recall Curves
        • Feature Importance + SHAP
```

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
|  **Logistic Regression** | 0.9960 | 0.9878 | 0.9926 | **0.9902** | **0.9998** |
| Gradient Boosting | 0.9965 | 0.9831 | 1.0000 | 0.9915 | 0.9997 |
| XGBoost | 0.9965 | 0.9831 | 1.0000 | 0.9915 | 0.9996 |
| Voting Ensemble | 0.9965 | 0.9831 | 1.0000 | 0.9915 | 0.9983 |
| Random Forest | 0.9965 | 0.9831 | 1.0000 | 0.9915 | 0.9981 |

> **Best model selected by ROC-AUC: Logistic Regression (0.9998)**

> ⚠️ Note: Extremely high scores are partly due to delay cause columns (`carrier_delay`, `weather_delay`, etc.) being present in the dataset. These columns are only populated for delayed flights, making them strong predictors. In a real-time deployment scenario, these would be excluded since they are not known before the flight.

---

## 📈 EDA Highlights

The pipeline generates 11 plots automatically:

- **Delay distribution** — histogram showing delay minutes + threshold line
- **Class balance** — pie chart of on-time vs delayed
- **Delay by hour** — peak delay times during the day
- **Delay by day of week** — which days have most delays
- **Delay by month & season** — seasonal patterns
- **Delay by distance** — short vs long haul comparison
- **Airline analysis** — delay rate per carrier + delay cause stacked bar
- **Route risk** — top 15 most delay-prone routes
- **Correlation heatmap** — feature relationships
- **ROC & PR curves** — all models compared
- **SHAP plots** — feature-level explainability

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. Copy each of the 23 cells from `Flight_Delay_Colab.ipynb` one by one
4. Run Cell 1 to install libraries
5. Run Cell 3 to upload your CSV file
6. Run cells sequentially — each builds on the previous

### Option 2 — Local (Python 3.8+)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/flight-delay-prediction.git
cd flight-delay-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your CSV in data/
# data/flight_data_2024_sample.csv

# 4. Run the pipeline
python flight_delay_prediction_final.py
```

---

## 🎯 Single Flight Prediction

After training, use the predictor to check any flight:

```python
predict_single_flight(
    airline     = 'WN',     # Southwest Airlines
    origin      = 'OMA',    # Omaha
    dest        = 'ATL',    # Atlanta
    dep_hour    = 17,       # 5:00 PM (peak hour)
    day_of_week = 5,        # Friday
    month       = 7,        # July (Summer)
    distance    = 1099.0
)
```

**Output:**
```
=============================================
  ✈️   OMA → ATL  |  17:00
  Airline     : WN
  Month       : 7  |  Day: 5
  Distance    : 1099 mi
---------------------------------------------
  Prediction  : 🔴 DELAYED
  Delay Prob  : 78.3%
  Risk Level  : 🔴 HIGH
=============================================
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `xgboost` | Gradient boosted trees |
| `imbalanced-learn` | SMOTE for class balancing |
| `shap` | Model explainability |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `joblib` | Model serialization |

---

## 📋 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
shap>=0.41.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- [ ] **Remove data leakage** — exclude delay cause columns for true pre-flight prediction
- [ ] **Real-time weather API** integration (OpenWeather / AviationWeather)
- [ ] **Regression model** — predict exact delay minutes, not just binary
- [ ] **Streamlit web app** — interactive UI for flight delay checker
- [ ] **Hyperparameter tuning** — full GridSearchCV on all models
- [ ] **Time-series validation** — train on Jan–Sep, test on Oct–Dec (more realistic split)
- [ ] **More data** — extend from 10K sample to full multi-year dataset
- [ ] **Airport congestion features** — number of flights per hour at origin

---

