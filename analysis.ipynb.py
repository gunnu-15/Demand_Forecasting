import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.set(style='whitegrid')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df = pd.read_csv("demand_forecasting.csv")
print(df.head(10))
# Convert Date from obj -> datetime
print(df.dtypes)
df['Date'] = pd.to_datetime(df['Date'])
print(df.info())

# DATA CLEANING

# Null/missing values, Duplicates, etc
print(df.isna().sum().sum())
print(df.duplicated().sum())

# Statistical Measures of dataset - Count, Min, Max, Mean, 255 Quartile, 505 Quartile, 75% Quartile, S.D
print(df.describe().T)
print(df.describe(include='object').T)

# Add Year, Month, Day, Weekday columns using Date column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.day_name()

# KPIs - Discounted Price, Sell Through Rate
df['Discounted Price'] = df['Price'] * (1 - df['Price'] / 100)
df['Sell Through Rate'] = df['Units Sold'] / df['Inventory Level']
print(df.head(1))
print(df['Sell Through Rate'].describe())

# Aggregate Demand by Category
print(df.groupby('Category')['Demand'].agg(['mean', 'sum', 'std']).sort_values(by = 'sum', ascending=False))

# Aggregate Demand by Region, Seasonality
print(df.groupby(['Region', 'Seasonality'])['Demand'].mean())
# Aggregate Demand by Promotion - Does running a promotion actually increase average demand?
print(df.groupby('Promotion')['Demand'].mean())

# EXPORATORY DATA ANALYSIS (EDA) — understanding your data's shape, distributions, and group differences before building any forecasting model

# Pivot Table using Pandas. Mean Monthly Demand by Category.
print(pd.pivot_table(df, values='Demand', index='Month', columns='Category', aggfunc='mean'))

# Seaborn's Histogram plot - Counts how many rows fall into each demand range (bucket/bin)
# Draw a smooth curve (Kernel Density Estimate) over the bars
# The distribution is RIGHT-SKEWED — a long tail toward 400, meaning high-demand events are rare but do happen
# Count - the number of rows (transactions/records) in your dataset that fall within each demand range
# The right skew (long tail) tells you extreme demand spikes exist but are rare — likely driven by promotions, seasons, or epidemics (columns you already have!)
# A skewed distribution like this often needs log transformation before feeding into ML models to perform well
sns.histplot(df['Demand'], bins=20, kde=True)
plt.title("Demand Distribution ")
plt.show()

# Scatter Plot - Units Sold vs Inventory
sns.scatterplot(data=df, x='Inventory Level', y='Units Sold')
plt.title('Inventory vs Units Sold')
plt.show()

# Box Plot -
sns.boxplot(data=df, x='Category', y='Demand', hue='Promotion', palette='Set2')
plt.xticks(rotation=45)
plt.title('Demand by Category')
plt.show()

# Box Plot -
sns.boxplot(data=df, x='Weather Condition', y='Demand', palette='Set1', hue='Weather Condition')
plt.xticks(rotation=45)
plt.title('Demand by Weather Condition')
plt.show()

# TIME SERIES ANALYSIS
monthly_demand = df.groupby('Month')['Demand'].mean()
monthly_demand.plot(kind='bar')
plt.title('Average/Mean Demand by Month')
plt.show()

daily_demand = df.groupby('Date')['Demand'].sum()
daily_demand.plot()
plt.title('Total Daily Demand over time')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()

sns.barplot(data=df, x='Promotion', y='Demand')
plt.title('Promotion Impact on Demand')
plt.show()