"""Generate demo datasets for AutoML-Lens testing."""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 300

# ── Classification Dataset ──────────────────────────────────
age = np.random.randint(18, 70, n)
income = np.random.normal(50000, 15000, n).astype(int)
credit_score = np.random.randint(300, 850, n)
years_employed = np.random.randint(0, 30, n)
num_products = np.random.randint(1, 5, n)
has_credit_card = np.random.choice([0, 1], n)
is_active = np.random.choice([0, 1], n)
balance = np.random.exponential(5000, n).round(2)

# Target: churn based on features
churn_prob = (
    0.1
    + 0.3 * (credit_score < 500).astype(float)
    + 0.2 * (years_employed < 2).astype(float)
    + 0.15 * (1 - is_active)
    - 0.1 * (num_products > 2).astype(float)
    + 0.1 * (age > 55).astype(float)
    + np.random.normal(0, 0.05, n)
)
churn = (churn_prob > 0.35).astype(int)

# Add some missing values
income_with_na = income.astype(float)
income_with_na[np.random.choice(n, 12, replace=False)] = np.nan
credit_with_na = credit_score.astype(float)
credit_with_na[np.random.choice(n, 8, replace=False)] = np.nan

geography = np.random.choice(["Urban", "Suburban", "Rural"], n, p=[0.5, 0.35, 0.15])
gender = np.random.choice(["Male", "Female"], n)

clf_df = pd.DataFrame({
    "CustomerID": range(1000, 1000 + n),
    "Age": age,
    "Gender": gender,
    "Geography": geography,
    "Income": income_with_na,
    "CreditScore": credit_with_na,
    "YearsEmployed": years_employed,
    "NumProducts": num_products,
    "HasCreditCard": has_credit_card,
    "IsActive": is_active,
    "Balance": balance,
    "Churn": churn,
})

script_dir = os.path.dirname(os.path.abspath(__file__))
clf_df.to_csv(os.path.join(script_dir, "classification.csv"), index=False)

# ── Regression Dataset ──────────────────────────────────────
sqft = np.random.randint(500, 5000, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 4, n)
house_age = np.random.randint(0, 50, n)
lot_size = np.random.uniform(0.1, 2.0, n).round(2)
garage_size = np.random.randint(0, 4, n)
neighborhood = np.random.choice(["Downtown", "Midtown", "Suburbs", "Rural", "Waterfront"], n)
condition = np.random.choice(["Excellent", "Good", "Fair", "Poor"], n, p=[0.2, 0.4, 0.3, 0.1])
has_pool = np.random.choice([0, 1], n, p=[0.7, 0.3])

# Price based on features
condition_map = {"Excellent": 1.2, "Good": 1.0, "Fair": 0.85, "Poor": 0.7}
neighborhood_map = {"Downtown": 1.3, "Midtown": 1.15, "Suburbs": 1.0, "Rural": 0.8, "Waterfront": 1.5}

price = (
    sqft * 150
    + bedrooms * 15000
    + bathrooms * 10000
    - house_age * 500
    + lot_size * 20000
    + garage_size * 8000
    + has_pool * 25000
    + np.array([neighborhood_map[n] for n in neighborhood]) * 50000
    + np.array([condition_map[c] for c in condition]) * 30000
    + np.random.normal(0, 15000, n)
).round(0).astype(int)

# Missing values
sqft_na = sqft.astype(float)
sqft_na[np.random.choice(n, 10, replace=False)] = np.nan
age_na = house_age.astype(float)
age_na[np.random.choice(n, 6, replace=False)] = np.nan

reg_df = pd.DataFrame({
    "PropertyID": range(5000, 5000 + n),
    "SquareFeet": sqft_na,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "HouseAge": age_na,
    "LotSize": lot_size,
    "GarageSize": garage_size,
    "Neighborhood": neighborhood,
    "Condition": condition,
    "HasPool": has_pool,
    "Price": price,
})

reg_df.to_csv(os.path.join(script_dir, "regression.csv"), index=False)

print(f"Created classification.csv ({len(clf_df)} rows, {len(clf_df.columns)} columns)")
print(f"Created regression.csv ({len(reg_df)} rows, {len(reg_df.columns)} columns)")
