import sqlite3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

DB_PATH = "data/hotel.db"
SEED = 42
FALSE_ALARM_COST = 500 # this can be configure for cost

CATEGORICAL = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type",
    "customer_type", "agent"
]

NUMERIC = [
    "lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes",
    "days_in_waiting_list", "adr", "required_car_parking_spaces",
    "total_of_special_requests"
]

# Load data
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql("SELECT * FROM bookings", conn)

# Basic cleaning
df = df.drop_duplicates(subset="booking_id").copy()

for col in CATEGORICAL + ["is_canceled"]:
    s = (
        df[col].astype(str).str.strip().str.lower()
        .str.replace(r"[\s\-]+", " ", regex=True)
    )
    df[col] = s.where(
        ~s.isin({"", "-", "na", "n/a", "nan", "null", "unknown", "undefined"})
    )

df["is_canceled"] = df["is_canceled"].map({
    "1": 1, "true": 1, "yes": 1,
    "0": 0, "false": 0, "no": 0
})

for col in NUMERIC:
    df[col] = pd.to_numeric(
        df[col].astype(str)
        .str.replace("SGD", "", regex=False)
        .str.replace("days", "", regex=False)
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

df.loc[(df["lead_time"] < 0) | (df["lead_time"] > 737), "lead_time"] = np.nan
df.loc[df["adr"] < 0, "adr"] = np.nan
df = df.dropna(subset=["is_canceled"])

# Prepare data
X = df[CATEGORICAL + NUMERIC]
y = df["is_canceled"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# Build and train model
categorical = Pipeline([
    ("fill", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50))
])

numeric = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cat", categorical, CATEGORICAL),
    ("num", numeric, NUMERIC)
])

model = Pipeline([
    ("encode", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

# Calculate booking value
booking_value = (
    df.loc[X_test.index, "adr"].fillna(0)
    * (
        df.loc[X_test.index, "stays_in_weekend_nights"].fillna(0)
        + df.loc[X_test.index, "stays_in_week_nights"].fillna(0)
    )
).to_numpy()

# Compare risk thresholds
results = []

for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:

    pred = (proba >= threshold).astype(int)

    missed = (y_test.to_numpy() == 1) & (pred == 0)
    false_alarms = (y_test.to_numpy() == 0) & (pred == 1)

    missed_loss = booking_value[missed].sum()
    false_alarm_cost = false_alarms.sum() * FALSE_ALARM_COST

    results.append({
        "Threshold": threshold,
        "Missed cancellations": int(missed.sum()),
        "False alarms": int(false_alarms.sum()),
        "Missed cancellation loss (S$)": round(missed_loss, 2),
        "False alarm cost (S$)": round(false_alarm_cost, 2),
        "Total cost (S$)": round(missed_loss + false_alarm_cost, 2)
    })

cost_results = pd.DataFrame(results)

print("\nThreshold Cost Comparison")
print(cost_results.to_string(index=False))

# Select lowest estimated cost
best = cost_results.loc[cost_results["Total cost (S$)"].idxmin()]

print("\nRecommended threshold:", best["Threshold"])
print("Minimum estimated cost: S$", f'{best["Total cost (S$)"]:,.2f}')

# Export results
cost_results.to_csv("cost_analysis_results.csv", index=False)
print("\nResults saved to cost_analysis_results.csv")