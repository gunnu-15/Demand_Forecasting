# FEATURE ENGINEERING & MACHINE LEARNING

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model for predicting our Demands
from xgboost import XGBRFRegressor

import pickle
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df = pd.read_csv('demand_forecasting.csv')
print(df)

print(df.columns)

features = ['Price', 'Discount', 'Inventory Level', 'Promotion', 'Competitor Pricing', 'Category']
print(features)

target = 'Demand'
X = df[features].copy()
print(X)
y = df[target]
print(y)

# Encode categorical features as ML modles don't accept strings(Category col in our dataset is a string).So, we use LabelEncoder for categorical columns
label_encoders = {}
categorical_cols = X.select_dtypes(include='object').columns
print(categorical_cols)

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
print(label_encoders)

# Split our data into training and testing sets, where training set has 80% of data & we're going to train our models with training set
# With testing set we're going to test our model's performance
# If we want to train a linear model, we need to do scaling because in X we have different sized values like
# InventoryLevel(in '00), Category(0,1), CompetitiorPricing(90's)

# XGBoost - Tree-based. In tree-based algorithms, we don't need scaling. We'll intentionally skip a StandardScaler to keep pipelines simpler & deployments friendly
# Tree-based models in real-life: Decision-Tree, Random Forest, extra trees, don't require scaling
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

xgb = XGBRFRegressor(objective='reg:squarederror', n_jobs = -1)
# After initializing our model, we'll do hyperparameter tuning bcoz base version of XGBoost won't give us highest performance
# Define dictionary for possible parameters & train model with possible parameters
param_dict = {
    'n_estimators': [200,300,500],
    'max_depth': [3,4,6,8],
    'learning_rate': [0.01,0.05,0.1],
    'subsample': [0.7,0.8,1.0],
    'colsample_bytree': [0.7,0.8,1.0],
    'min_child_weight': [1,3,5]
}

random_search = RandomizedSearchCV(
    estimator = xgb,
    param_distributions = param_dict,
    n_iter = 25,
    scoring = 'neg_mean_absolute_error',
    cv = 3,
    verbose = 1,
    n_jobs = -1
)

# This will try random combinations of values of param_dict, and will return the best result, in terms of neg_mean_absolute_error
random_search.fit(X_train, y_train)

# This'll show us the selected parameters best performing
random_search.best_params_

# Now, we can make predictions with this model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mean_squared_error(y_test, y_pred)

# Returns an array of importance of columns one-by-one on predictions: whihc column is most effective on the result
feature_importance = pd.Series(
    best_model.feature_importances_,
    index = X.columns
).sort_values(ascending=False)

print(feature_importance)

feature_importance.plot(kind='bar', title='Feature Importances')

# Encoder ready
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Model ready
with open('xgboost_demand_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)