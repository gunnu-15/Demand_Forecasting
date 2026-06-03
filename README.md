# 📦 Demand Forecasting App

An end-to-end Machine Learning project that predicts product demand using XGBoost, built with Python and deployed via a Streamlit web interface.

---

## 🚀 Live Demo

Run locally with:
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Demand_Forecasting/
│
├── analysis.ipynb.py        # Data cleaning, EDA, visualizations
├── machine_learning.py      # Feature engineering, model training & evaluation
├── app.py                   # Streamlit web app for demand prediction
├── demand_forecasting.csv   # Dataset
├── xgboost_demand_model.pkl # Trained XGBoost model
├── label_encoders.pkl       # Saved label encoders for categorical features
└── README.md
```

---

## 📊 Dataset

The dataset (`demand_forecasting.csv`) contains retail records with the following features:

| Column | Description |
|---|---|
| Date | Transaction date |
| Store ID | Store identifier |
| Product ID | Product identifier |
| Category | Product category (Electronics, Clothing, Groceries, Toys, Furniture) |
| Region | Store region (North, South, East, West) |
| Inventory Level | Current stock level |
| Units Sold | Units sold on that day |
| Units Ordered | Units ordered for restocking |
| Price | Product price |
| Discount | Discount percentage applied |
| Weather Condition | Weather on that day (Snowy, Rainy, Sunny, etc.) |
| Promotion | Whether a promotion was active (0 or 1) |
| Competitor Pricing | Competitor's price for the same product |
| Seasonality | Season (Winter, Summer, etc.) |
| Demand | **Target variable** — actual demand units |

---

## 🔍 Exploratory Data Analysis (`analysis.ipynb.py`)

- Null/duplicate checks
- Feature extraction from `Date` → Year, Month, Day, Weekday
- KPI engineering: **Discounted Price**, **Sell Through Rate**
- Aggregations: Demand by Category, Region, Seasonality, Promotion
- Pivot table: Mean monthly demand by category
- Visualizations:
  - Demand distribution histogram (with KDE)
  - Inventory vs Units Sold scatter plot
  - Demand by Category & Weather Condition box plots
  - Monthly and daily demand time series
  - Promotion impact bar chart

---

## 🤖 Machine Learning (`machine_learning.py`)

### Features Used
```
Price, Discount, Inventory Level, Promotion, Competitor Pricing, Category
```

### Pipeline
1. **Label Encoding** — Categorical columns encoded with `LabelEncoder`
2. **Train/Test Split** — 80% train, 20% test
3. **Model** — `XGBRFRegressor` (XGBoost Random Forest Regressor)
4. **Hyperparameter Tuning** — `RandomizedSearchCV` with 25 iterations, 3-fold CV
5. **Evaluation** — MAE, MSE, R² Score
6. **Feature Importance** — Ranked by impact on predictions

### Why XGBoost (tree-based)?
Tree-based models don't require feature scaling, making the pipeline simpler and deployment-friendly.

### Saved Artifacts
- `xgboost_demand_model.pkl` — Best trained model
- `label_encoders.pkl` — Encoders for inference

---

## 🌐 Streamlit App (`app.py`)

Interactive web UI for real-time demand prediction.

**Input fields:**
- Price
- Discount
- Inventory Level
- Promotion (Yes/No)
- Competitor Pricing
- Category

**Output:**
> ✅ Predicted Demand: _X_ Units

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/gunnu-15/Demand_Forecasting.git
cd Demand_Forecasting

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
streamlit
pickle-mixin
```

> Generate `requirements.txt` with: `pip freeze > requirements.txt`

---

## 📈 Results

- Model: XGBoost Random Forest Regressor
- Tuning: RandomizedSearchCV (25 iterations, 3-fold CV)
- Metric: Negative Mean Absolute Error (scoring)

---

## 👤 Author

**Jannu Akash**
GitHub: [@gunnu-15](https://github.com/gunnu-15)
